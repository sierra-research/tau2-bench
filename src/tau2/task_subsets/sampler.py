# Copyright Sierra
"""Seeded stratified draw that produces a :class:`TaskSubset`.

The design in one line: partition the domain's task set into strata that name
the things a subset must not accidentally drop, allocate the 50 slots across
them in proportion to the frame, and draw inside each stratum with a recorded
seed.

Strata are declared per domain from a CLOSED catalog below — not inferred, not
passed in as free text — so a subset artifact's ``stratifier`` field always
names something that still exists in code and can be recomputed.
"""

import datetime
import random
from typing import Callable

from tau2.config import DEFAULT_TASK_SUBSET_SEED, DEFAULT_TASK_SUBSET_SIZE
from tau2.data_model.tasks import Task
from tau2.task_subsets.schema import (
    StratumRecord,
    SubsetFrame,
    SubsetStrategy,
    TaskSubset,
    frame_digest,
)
from tau2.utils.utils import get_commit_hash

# A stratifier maps the frame to one stratum key per task id. It takes the
# whole frame (not one task) so it can load domain metadata — e.g. which tools
# mutate state — exactly once.
StratifierFn = Callable[[str, list[Task]], dict[str, str]]


# ---------------------------------------------------------------------------
# Stratifiers
# ---------------------------------------------------------------------------


def _telecom_issue_persona(domain: str, tasks: list[Task]) -> dict[str, str]:
    """Telecom: scenario family x persona difficulty, both read off the id.

    Telecom ids are generated, not authored:
    ``[mms_issue]airplane_mode_on|bad_vpn[PERSONA:Hard]``. The family is the
    bracketed prefix and the difficulty is the ``PERSONA:`` tag; the structured
    ``user_scenario.persona`` is the rendered prose, which carries the same
    difficulty but only as free text.

    Family is the axis a prefix cut destroys — the file is grouped by it — and
    persona difficulty is the axis a family-only draw would leave to chance.
    """
    keys: dict[str, str] = {}
    for task in tasks:
        family = "unknown"
        if task.id.startswith("[") and "]" in task.id:
            family = task.id[1 : task.id.index("]")]
        persona = "none"
        if "[PERSONA:" in task.id:
            persona = task.id.rsplit("[PERSONA:", 1)[1].split("]", 1)[0].lower()
        keys[task.id] = f"{family}/{persona}"
    return keys


def _write_profile(domain: str, tasks: list[Task]) -> dict[str, str]:
    """Airline, retail: primary write action x how many writes the task needs.

    These domains have opaque ids ("0".."113"), so the stratum comes from the
    task's own evaluation criteria. The primary write action is what the task
    is ABOUT (an exchange, a cancellation, a pure lookup), and the write count
    is a difficulty proxy — a one-write task and a four-write task exercise
    very different amounts of policy.

    Read/write is taken from the domain's toolkit rather than from tool-name
    prefixes, so it stays right when a domain adds a tool.
    """
    write_tools = _write_tool_names(domain)
    keys: dict[str, str] = {}
    for task in tasks:
        actions = []
        if task.evaluation_criteria is not None:
            actions = list(task.evaluation_criteria.actions or [])
        writes = [a.name for a in actions if a.name in write_tools]
        primary = writes[0] if writes else "read_only"
        count = len(writes)
        load = "3+" if count >= 3 else str(count)
        keys[task.id] = f"{primary}/{load}"
    return keys


def _write_tool_names(domain: str) -> set[str]:
    """Names of the domain's state-mutating tools, from its toolkit."""
    from tau2.environment.toolkit import ToolType
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)()
    toolkit = env.tools
    return {
        name
        for name in toolkit.tools
        if toolkit.tool_type(name) == ToolType.WRITE  # type: ignore[union-attr]
    }


def _uniform(domain: str, tasks: list[Task]) -> dict[str, str]:
    """One stratum: a plain seeded uniform draw, recorded as such.

    For domains whose tasks carry no axis the other stratifiers can read —
    banking_knowledge's write actions are USER-requestor tool calls, so
    ``write_profile`` would file nearly every task under ``read_only`` and
    stratification would be decoration pretending to be design.
    """
    return {task.id: "all" for task in tasks}


STRATIFIERS: dict[str, StratifierFn] = {
    "issue_persona": _telecom_issue_persona,
    "write_profile": _write_profile,
    "uniform": _uniform,
}

DOMAIN_STRATIFIER: dict[str, str] = {
    "telecom": "issue_persona",
    "telecom-workflow": "issue_persona",
    "airline": "write_profile",
    "retail": "write_profile",
    "banking_knowledge": "uniform",
}

DEFAULT_STRATIFIER = "write_profile"


def stratifier_for(domain: str) -> str:
    return DOMAIN_STRATIFIER.get(domain, DEFAULT_STRATIFIER)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def allocate(frame_counts: dict[str, int], size: int) -> dict[str, int]:
    """Slots per stratum: proportional, largest-remainder, min one each.

    Three rules, in order of precedence:

    1. Never draw more from a stratum than it holds.
    2. Every non-empty stratum gets at least one slot, if the budget can pay
       for it. Coverage is the reason to stratify at all; a proportional rule
       alone rounds small strata to zero, which is the failure this whole
       module exists to fix.
    3. Otherwise proportional to frame size, remainders to the largest
       fractional part (ties by stratum key, so the result is deterministic).
    """
    total = sum(frame_counts.values())
    if total == 0:
        return {}
    if size >= total:
        return dict(frame_counts)

    keys = sorted(frame_counts)
    exact = {k: size * frame_counts[k] / total for k in keys}
    alloc = {k: min(int(exact[k]), frame_counts[k]) for k in keys}

    # Remainders, largest fractional part first.
    remaining = size - sum(alloc.values())
    by_fraction = sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k))
    while remaining > 0:
        progressed = False
        for key in by_fraction:
            if remaining == 0:
                break
            if alloc[key] < frame_counts[key]:
                alloc[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # every stratum is at capacity
            break

    # Min-one coverage: pay for empty strata out of the most over-served ones.
    empty = [k for k in keys if alloc[k] == 0 and frame_counts[k] > 0]
    for key in empty:
        donor = max(
            (k for k in keys if alloc[k] > 1),
            key=lambda k: (alloc[k] / frame_counts[k], alloc[k], k),
            default=None,
        )
        if donor is None:  # budget smaller than the number of strata
            break
        alloc[donor] -= 1
        alloc[key] = 1
    return alloc


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------


def build_prefix_subset(
    *,
    name: str,
    parent: TaskSubset,
    tasks: list[Task],
    size: int,
) -> TaskSubset:
    """The first ``size`` ids of ``parent``, frozen as a subset of its own.

    A DERIVED artifact, not a new draw: the ids are ``parent.task_ids[:size]``
    in the parent's recorded order, so a cell run on it pairs exactly with the
    parent's cells restricted to the same stems. ``tasks`` must be the
    parent's own frame (same task set, same split) — the frame digest is
    checked so a prefix can never be cut from a drifted frame — and is used to
    recompute the parent stratifier's per-stratum accounting over the kept ids
    (strata whose members all fall outside the prefix record ``drawn=0``).
    """
    if size <= 0 or size >= parent.size:
        raise ValueError(
            f"prefix size must be in [1, {parent.size - 1}] for parent "
            f"'{parent.name}' ({parent.size} ids); got {size}"
        )
    frame_ids = [t.id for t in tasks]
    if frame_digest(frame_ids) != parent.frame.digest:
        raise ValueError(
            f"frame of task set '{parent.frame.task_set}' no longer matches "
            f"subset '{parent.name}' — a prefix cut from a drifted frame would "
            "not pair with the parent's runs. Redraw the parent first."
        )
    kept = parent.task_ids[:size]
    if parent.stratifier in STRATIFIERS:
        keys = STRATIFIERS[parent.stratifier](parent.domain, tasks)
        counts: dict[str, int] = {}
        drawn: dict[str, int] = {}
        for task_id in frame_ids:
            counts[keys[task_id]] = counts.get(keys[task_id], 0) + 1
        for task_id in kept:
            drawn[keys[task_id]] = drawn.get(keys[task_id], 0) + 1
        strata = [
            StratumRecord(key=key, frame_size=counts[key], drawn=drawn.get(key, 0))
            for key in sorted(counts)
        ]
    else:  # parent was a census ('none') — one accounting stratum
        strata = [StratumRecord(key="all", frame_size=len(frame_ids), drawn=len(kept))]
    return TaskSubset(
        name=name,
        domain=parent.domain,
        strategy=SubsetStrategy.PREFIX_OF,
        stratifier=parent.stratifier,
        seed=parent.seed,
        frame=parent.frame,
        strata=strata,
        created=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        git_commit=get_commit_hash(),
        derived_from=parent.name,
        task_ids=kept,
    )


def build_subset(
    *,
    name: str,
    domain: str,
    task_set: str,
    tasks: list[Task],
    task_split: str | None = "base",
    size: int = DEFAULT_TASK_SUBSET_SIZE,
    seed: int = DEFAULT_TASK_SUBSET_SEED,
    stratifier: str | None = None,
) -> TaskSubset:
    """Draw ``size`` tasks from ``tasks`` and return the artifact.

    Frames no larger than ``size`` are recorded as a census: the artifact still
    exists (a run naming the subset gets the same pinned list either way), it
    just has nothing to sample.
    """
    if not tasks:
        raise ValueError(f"Task set '{task_set}' is empty; nothing to sample")
    frame_ids = [t.id for t in tasks]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"Task set '{task_set}' has duplicate task ids")

    frame = SubsetFrame(
        task_set=task_set,
        task_split=task_split,
        size=len(tasks),
        digest=frame_digest(frame_ids),
    )
    created = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    if size >= len(tasks):
        return TaskSubset(
            name=name,
            domain=domain,
            strategy=SubsetStrategy.CENSUS,
            stratifier="none",
            seed=seed,
            frame=frame,
            strata=[StratumRecord(key="all", frame_size=len(tasks), drawn=len(tasks))],
            created=created,
            git_commit=get_commit_hash(),
            task_ids=frame_ids,
        )

    stratifier_name = stratifier or stratifier_for(domain)
    if stratifier_name not in STRATIFIERS:
        raise ValueError(
            f"Unknown stratifier '{stratifier_name}'. Known: {sorted(STRATIFIERS)}"
        )
    keys = STRATIFIERS[stratifier_name](domain, tasks)

    # Frame order inside each stratum, so the draw is reproducible from the
    # file alone and does not depend on dict iteration order.
    members: dict[str, list[str]] = {}
    for task in tasks:
        members.setdefault(keys[task.id], []).append(task.id)
    counts = {k: len(v) for k, v in members.items()}
    alloc = allocate(counts, size)

    drawn: set[str] = set()
    for key in sorted(members):
        k = alloc.get(key, 0)
        if k == 0:
            continue
        # One independent stream per stratum: re-drawing one stratum (a new
        # size, a changed frame) leaves the others' draws untouched.
        rng = random.Random(f"{seed}|{domain}|{name}|{key}")
        drawn.update(rng.sample(members[key], k))

    task_ids = [tid for tid in frame_ids if tid in drawn]
    strata = [
        StratumRecord(key=key, frame_size=counts[key], drawn=alloc.get(key, 0))
        for key in sorted(members)
    ]
    return TaskSubset(
        name=name,
        domain=domain,
        strategy=SubsetStrategy.STRATIFIED,
        stratifier=stratifier_name,
        seed=seed,
        frame=frame,
        strata=strata,
        created=created,
        git_commit=get_commit_hash(),
        task_ids=task_ids,
    )
