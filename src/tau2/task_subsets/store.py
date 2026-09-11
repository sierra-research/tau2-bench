# Copyright Sierra
"""Reading, writing and resolving the checked-in subset artifacts.

Artifacts live at ``data/tau2/task_subsets/<name>.json`` — one file per subset,
per domain, never pooled across domains.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.config import CANONICAL_TASK_SUBSETS, SUBSET_ALL, SUBSET_AUTO
from tau2.task_subsets.schema import TaskSubset, frame_digest


def task_subset_dir() -> Path:
    """The CURRENT ``data/tau2/task_subsets`` directory.

    Resolved dynamically, never snapshotted at import time, so an isolated
    TAU2_DATA_DIR (tests, a vendored data tree) is honoured.
    """
    import tau2.utils

    return tau2.utils.DATA_DIR / "tau2" / "task_subsets"


# SUBSET_AUTO: use the domain's canonical subset, if it has one.
# SUBSET_ALL: run the whole task set.
__all__ = [
    "SUBSET_AUTO",
    "SUBSET_ALL",
    "task_subset_dir",
    "apply_subset",
    "canonical_subset_name",
    "check_frame",
    "list_subsets",
    "load_subset",
    "resolve_subset",
    "save_subset",
    "subset_covering_ids",
    "subset_path",
]


def subset_path(name: str) -> Path:
    return task_subset_dir() / f"{name}.json"


def list_subsets() -> list[str]:
    directory = task_subset_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def load_subset(name: str) -> TaskSubset:
    path = subset_path(name)
    if not path.is_file():
        known = list_subsets()
        raise FileNotFoundError(
            f"No task subset '{name}' at {path}. "
            f"Known subsets: {known or '(none)'}. "
            f"Build one with `tau2 tasks subset --domain <domain>`."
        )
    subset = TaskSubset.model_validate(json.loads(path.read_text()))
    if subset.name != name:
        raise ValueError(
            f"Task subset file {path} declares name '{subset.name}'; "
            "the filename stem is the subset's identity and must match."
        )
    return subset


def save_subset(subset: TaskSubset, *, overwrite: bool = False) -> Path:
    path = subset_path(subset.name)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. A subset is frozen once runs have scored "
            "on it — publish a new name (e.g. '<domain>_50_v2') rather than "
            "silently changing this one, or pass --overwrite if it has never "
            "been run."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset.model_dump(mode="json"), indent=2) + "\n")
    return path


def canonical_subset_name(domain: str) -> Optional[str]:
    """The subset a run on ``domain`` uses when --task-subset is 'auto'."""
    return CANONICAL_TASK_SUBSETS.get(domain)


def resolve_subset(
    *,
    domain: str,
    requested: Optional[str],
    explicit_task_selection: bool = False,
) -> Optional[TaskSubset]:
    """The subset a run should apply, or None for the whole task set.

    ``requested`` is the ``--task-subset`` value: a name, ``auto``, ``all``,
    or None (treated as ``auto``). An explicit ``--task-ids``/``--num-tasks``
    beats ``auto`` — those flags exist to say "run exactly these", and a
    default that quietly intersected with them would make short runs look like
    subset runs. A NAMED subset is never overridden that way; it errors.
    """
    requested = (requested or SUBSET_AUTO).strip()
    if requested == SUBSET_ALL:
        return None
    if requested == SUBSET_AUTO:
        if explicit_task_selection:
            return None
        name = canonical_subset_name(domain)
        if name is None:
            return None
        try:
            subset = load_subset(name)
        except FileNotFoundError as exc:
            # An isolated TAU2_DATA_DIR (tests, a vendored data tree) may not
            # carry the artifact. Degrade to the whole task set rather than
            # refusing to run — but say so, because the run is then scoring on
            # a different population than the one the domain's numbers assume.
            logger.warning(
                f"Domain '{domain}' names canonical subset '{name}', which is "
                f"not in this data directory — running the WHOLE task set. {exc}"
            )
            return None
    else:
        subset = load_subset(requested)
        if explicit_task_selection:
            raise ValueError(
                f"--task-subset {requested} cannot be combined with explicit "
                "--task-ids/--num-tasks: the subset IS the task selection. "
                "Drop one, or pass --task-subset all."
            )
    if subset.domain != domain and canonical_subset_name(domain) != subset.name:
        # The subset was drawn for another domain and this domain does not claim
        # it either. Two domains may deliberately share one draw when they are
        # the same task pool under different policies (telecom /
        # telecom-workflow) — that is what naming it in CANONICAL_TASK_SUBSETS
        # says. Anything else is a frame mismatch dressed up as a filter.
        raise ValueError(
            f"Task subset '{subset.name}' belongs to domain '{subset.domain}', "
            f"but this run is on '{domain}', which does not declare it in "
            "CANONICAL_TASK_SUBSETS. Subsets are per domain and are shared only "
            "between domains that are explicitly the same task pool."
        )
    return subset


def subset_covering_ids(*, domain: str, task_ids: list[str]) -> Optional[TaskSubset]:
    """The domain's canonical subset, iff every id in ``task_ids`` belongs to it.

    For PROVENANCE only, never filtering: a run that enumerates its tasks with
    --task-ids may still be scoring the subset, which is exactly what the
    multilingual presets do (they apply the subset when building the arm, so
    the subset default is switched off by the time `tau2 run` sees the list).
    Without this, `tau2 run-preset` results — the ones the paper quotes — would
    be the only ones that never say which population they scored.

    Returns None as soon as an id strays outside the subset, so an unrelated
    --task-ids run is not labelled with a subset it does not belong to.
    """
    name = canonical_subset_name(domain)
    if name is None or not task_ids:
        return None
    try:
        subset = load_subset(name)
    except FileNotFoundError:
        return None
    if all(subset.matches(task_id) is not None for task_id in task_ids):
        return subset
    return None


def check_frame(subset: TaskSubset, frame_task_ids: list[str]) -> Optional[str]:
    """Warn text if the frame has drifted since the subset was drawn, else None.

    Only meaningful against the frame's OWN task set: a localized set is a
    rename of the frame, so its ids never match the recorded digest.
    """
    if frame_digest(frame_task_ids) == subset.frame.digest:
        return None
    return (
        f"Task subset '{subset.name}' was drawn from a {subset.frame.size}-task "
        f"frame whose ids no longer match task set '{subset.frame.task_set}' "
        f"(now {len(frame_task_ids)} tasks). The pinned ids still run, but the "
        "sampling design no longer describes the current frame — redraw as a "
        "new subset name if the frame change was intended."
    )


def apply_subset(subset: TaskSubset, tasks: list, task_set_name: str) -> list:
    """Filter ``tasks`` to the subset, logging frame drift where detectable."""
    # An alias task set that claims this subset as canonical (telecom-workflow
    # registers telecom's own loader) IS the frame under another name, so drift
    # is still detectable there — a localized rename is not.
    if task_set_name == subset.frame.task_set or (
        canonical_subset_name(task_set_name) == subset.name
    ):
        drift = check_frame(subset, [t.id for t in tasks])
        if drift is not None:
            logger.warning(drift)
    return subset.select(tasks)
