"""Source-derived strata and the donor's deterministic largest-remainder draw."""

import random
from collections import Counter
from datetime import datetime, timezone

from tau2.task_subsets.schema import (
    CanonicalFrame,
    DrawRequest,
    StratumRecord,
    SubsetFrame,
    TaskSubset,
    frame_digest,
)


def _strata(frame: CanonicalFrame) -> dict[str, str]:
    if frame.domain == "telecom":
        result = {}
        for task in frame.tasks:
            if not task.id.startswith("[") or "]" not in task.id:
                raise ValueError("telecom task lacks a scenario family")
            family = task.id[1 : task.id.index("]")]
            persona = "none"
            if "[PERSONA:" in task.id:
                persona = task.id.rsplit("[PERSONA:", 1)[1].split("]", 1)[0].lower()
            result[task.id] = f"{family}/{persona}"
        return result
    from tau2.environment.toolkit import ToolType
    from tau2.registry import registry

    toolkit = registry.get_env_constructor(frame.domain)().tools
    write_tools = {
        name for name in toolkit.tools if toolkit.tool_type(name) == ToolType.WRITE
    }
    result = {}
    for task in frame.tasks:
        actions = (
            task.evaluation_criteria.actions or [] if task.evaluation_criteria else []
        )
        writes = [action.name for action in actions if action.name in write_tools]
        count = "3+" if len(writes) >= 3 else str(len(writes))
        result[task.id] = f"{writes[0] if writes else 'read_only'}/{count}"
    return result


def _allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    total = sum(counts.values())
    if size >= total:
        return dict(counts)
    keys = sorted(counts)
    exact = {key: size * counts[key] / total for key in keys}
    allocation = {key: min(int(exact[key]), counts[key]) for key in keys}
    remaining = size - sum(allocation.values())
    for key in sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k)):
        if remaining and allocation[key] < counts[key]:
            allocation[key] += 1
            remaining -= 1
    for key in [key for key in keys if allocation[key] == 0]:
        donor = max(
            (k for k in keys if allocation[k] > 1),
            key=lambda k: (allocation[k] / counts[k], allocation[k], k),
            default=None,
        )
        if donor is None:
            break
        allocation[donor] -= 1
        allocation[key] = 1
    return allocation


def build_subset(
    request: DrawRequest,
    frame: CanonicalFrame,
    *,
    created: str | None = None,
    git_commit: str | None = None,
) -> TaskSubset:
    """Reproduce the reviewed seeded draw; publication is a separate operation."""
    if request.domain != frame.domain:
        raise ValueError("draw domain disagrees with canonical frame")
    frame = CanonicalFrame.model_validate(frame.model_dump())
    census = request.size >= len(frame.tasks)
    keys = {task.id: "all" for task in frame.tasks} if census else _strata(frame)
    counts = dict(Counter(keys.values()))
    allocation = _allocate(counts, request.size)
    selected = set()
    for key in sorted(counts):
        members = [task.id for task in frame.tasks if keys[task.id] == key]
        rng = random.Random(f"{request.seed}|{frame.domain}|{request.name}|{key}")
        selected.update(rng.sample(members, allocation[key]))
    return TaskSubset(
        name=request.name,
        domain=frame.domain,
        strategy="census" if census else "stratified",
        stratifier="none"
        if census
        else "issue_persona"
        if frame.domain == "telecom"
        else "write_profile",
        seed=request.seed,
        frame=SubsetFrame(
            task_set=frame.domain,
            task_split=frame.task_split,
            size=len(frame.tasks),
            digest=frame_digest(frame.task_ids),
        ),
        strata=[
            StratumRecord(key=key, frame_size=counts[key], drawn=allocation[key])
            for key in sorted(counts)
        ],
        created=created or datetime.now(timezone.utc).date().isoformat(),
        git_commit=git_commit,
        task_ids=[task_id for task_id in frame.task_ids if task_id in selected],
    )
