# Copyright Sierra
"""Typed schema for a fixed task subset — the frozen frame a run scores on.

A subset is a checked-in artifact under ``data/tau2/task_subsets/``: the domain
it belongs to, the frame it was drawn from, the design it was drawn under, and
the exact task ids that came out. Runs name it (``--task-subset telecom_50``)
rather than re-deriving it, so "the 50 tasks" is one reviewable file instead of
a ``--num-tasks 50`` that silently means "whatever the file lists first".

Subsets are PER DOMAIN and never pooled: airline's 50 and telecom's 50 are two
independent draws from two frames, and a run picks the one for its domain.
"""

import hashlib
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.tasks import Task


class SubsetStrategy(str, Enum):
    """How the ids were chosen."""

    CENSUS = "census"
    """The frame is no larger than the target size: every task is in."""

    STRATIFIED = "stratified"
    """Proportional allocation over strata, seeded draw within each."""

    PREFIX_OF = "prefix_of"
    """The first N ids of an existing subset, in its recorded (frame) order.

    A DERIVED subset: no new draw happens. It exists so a smaller cell can
    pair exactly with the parent's cells restricted to the same stems —
    ``derived_from`` names the parent, and the strata are the parent's
    stratifier recomputed over the kept ids (accounting, not design)."""


def frame_digest(task_ids: list[str]) -> str:
    """Digest of a frame's task ids, in file order.

    Pins WHICH frame the draw was made from. If a domain's task file gains,
    loses, or reorders tasks, the recorded digest stops matching and the
    subset is stale — a silently-shifted frame would otherwise turn a fixed
    subset back into an arbitrary one.
    """
    joined = "\n".join(task_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


class SubsetFrame(BaseModel):
    """The population the subset was drawn from."""

    model_config = ConfigDict(extra="forbid")

    task_set: Annotated[
        str,
        Field(description="Registered task set the frame was loaded from."),
    ]
    task_split: Annotated[
        Optional[str],
        Field(
            default="base",
            description="Split the frame was loaded with. Load-bearing, not "
            "decoration: `telecom` yields 114 tasks on the 'base' split and "
            "2285 without one, so a frame that omits the split cannot be "
            "reloaded.",
        ),
    ]
    size: Annotated[int, Field(ge=1, description="Number of tasks in the frame.")]
    digest: Annotated[
        str,
        Field(description="sha256 of the frame's task ids in file order."),
    ]


class StratumRecord(BaseModel):
    """One stratum's accounting: how big it was, how many it contributed.

    Kept per stratum rather than as a single overall rate because the whole
    point of stratifying is that the strata are not interchangeable — the
    inclusion probability that puts an estimate back on the population scale
    is this one, not 50/N.
    """

    model_config = ConfigDict(extra="forbid")

    key: Annotated[str, Field(description="Stratum label from the stratifier.")]
    frame_size: Annotated[int, Field(ge=1, description="Tasks in this stratum.")]
    drawn: Annotated[int, Field(ge=0, description="Tasks drawn from it.")]

    @property
    def inclusion_probability(self) -> float:
        return self.drawn / self.frame_size


class TaskSubset(BaseModel):
    """A named, frozen set of task ids for one domain."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str,
        Field(description="Subset name; the artifact's filename stem."),
    ]
    domain: Annotated[
        str,
        Field(
            description="Domain the subset belongs to. A run whose domain "
            "differs is refused rather than silently filtered to nothing."
        ),
    ]
    strategy: Annotated[SubsetStrategy, Field(description="How ids were chosen.")]
    stratifier: Annotated[
        str,
        Field(
            description="Name of the stratifier used (see "
            "tau2.task_subsets.sampler.STRATIFIERS); 'none' for a census."
        ),
    ]
    seed: Annotated[
        int,
        Field(description="Seed for the within-stratum draw. Unused on a census."),
    ]
    frame: Annotated[SubsetFrame, Field(description="Population drawn from.")]
    strata: Annotated[
        list[StratumRecord],
        Field(description="Per-stratum accounting, in stratum-key order."),
    ]
    created: Annotated[
        str, Field(description="ISO date the artifact was generated (UTC).")
    ]
    git_commit: Annotated[
        Optional[str],
        Field(default=None, description="Repo commit the artifact was built at."),
    ]
    derived_from: Annotated[
        Optional[str],
        Field(
            default=None,
            description="For PREFIX_OF subsets: the parent subset this one is "
            "the prefix of. None for direct draws.",
        ),
    ]
    task_ids: Annotated[
        list[str],
        Field(
            description="The drawn ids, in FRAME order. Canonical (unlocalized) "
            "ids: a localized set's '<id>_<lang>' variants resolve through "
            "select()."
        ),
    ]

    @property
    def size(self) -> int:
        """Derived, never stored — a stored size can disagree with the list."""
        return len(self.task_ids)

    @model_validator(mode="after")
    def _check_consistency(self) -> "TaskSubset":
        if not self.task_ids:
            raise ValueError(f"Task subset '{self.name}' has no task ids")
        if len(set(self.task_ids)) != len(self.task_ids):
            dupes = sorted({t for t in self.task_ids if self.task_ids.count(t) > 1})
            raise ValueError(f"Task subset '{self.name}' repeats ids: {dupes}")
        drawn = sum(s.drawn for s in self.strata)
        if drawn != len(self.task_ids):
            raise ValueError(
                f"Task subset '{self.name}': strata account for {drawn} tasks "
                f"but {len(self.task_ids)} ids are listed"
            )
        if len(self.task_ids) > self.frame.size:
            raise ValueError(
                f"Task subset '{self.name}' draws {len(self.task_ids)} tasks "
                f"from a frame of {self.frame.size}"
            )
        return self

    def matches(self, task_id: str) -> Optional[str]:
        """The canonical id this task id belongs to, or None.

        Localized task sets rename every task by appending a suffix to the
        canonical id (``0`` -> ``0_es`` -> ``0_es_identity``), so membership is
        "is the canonical id, or extends it at a separator". Prefix matching is
        unambiguous because the separator is required: ``0_es`` cannot match
        canonical ``0_e``, and ``10_es`` cannot match canonical ``1``.
        """
        for canonical in self.task_ids:
            if task_id == canonical or task_id.startswith(f"{canonical}_"):
                return canonical
        return None

    def select_ids(self, task_ids: list[str]) -> list[str]:
        """The subset's ids as they appear in ``task_ids``, in subset order.

        Raises if a canonical id matches nothing: a localized set that is
        missing tasks would otherwise quietly run a short subset, and a
        47-task "50-task run" is the kind of thing that is discovered after
        the numbers are in a paper.
        """
        by_canonical: dict[str, list[str]] = {}
        for task_id in task_ids:
            canonical = self.matches(task_id)
            if canonical is not None:
                by_canonical.setdefault(canonical, []).append(task_id)

        missing = [tid for tid in self.task_ids if tid not in by_canonical]
        if missing:
            raise ValueError(
                f"Task subset '{self.name}' does not fit this task set: "
                f"{len(missing)} of {self.size} ids matched no task "
                f"(first missing: {missing[0]!r})"
            )
        ambiguous = {k: v for k, v in by_canonical.items() if len(v) > 1}
        if ambiguous:
            key, ids = next(iter(ambiguous.items()))
            raise ValueError(
                f"Task subset '{self.name}': canonical id {key!r} matches "
                f"{len(ids)} tasks in this task set ({ids[:3]}). The task set "
                "mixes localized variants that the subset cannot disambiguate."
            )
        return [by_canonical[tid][0] for tid in self.task_ids]

    def intersect_ids(self, task_ids: list[str]) -> list[str]:
        """Like :meth:`select_ids`, but tolerates a task set that falls short.

        For callers that must not hard-fail on an incomplete task set — preset
        GENERATION, which builds the table for every registered language at
        once, so one partially-localized pack cannot be allowed to take the
        others down. Everything that actually runs tasks uses the strict path.
        """
        by_canonical: dict[str, str] = {}
        for task_id in task_ids:
            canonical = self.matches(task_id)
            if canonical is not None:
                by_canonical.setdefault(canonical, task_id)
        return [by_canonical[tid] for tid in self.task_ids if tid in by_canonical]

    def select(self, tasks: list[Task]) -> list[Task]:
        """The subset's tasks, in subset order."""
        by_id = {task.id: task for task in tasks}
        if len(by_id) != len(tasks):
            raise ValueError(
                f"Task subset '{self.name}': the task set has duplicate ids"
            )
        return [by_id[tid] for tid in self.select_ids(list(by_id))]
