# Copyright Sierra
"""Environment, task loading, and splits for the intake domain."""

from typing import Optional

from pydantic import ValidationError

from tau2.data_model.tasks import Task
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.tools import IntakeTools
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.domains.intake.utils import (
    INTAKE_DATA_DIR,
    INTAKE_DB_PATH,
    INTAKE_MAIN_POLICY_PATH,
    INTAKE_TASK_SET_PATH,
    INTAKE_TEXT_POLICY_PATH,
    INTAKE_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

# Agent policy per run channel. The voice policy scripts spoken-channel
# mechanics — spell-out pinning, per-entity read-backs, capture logging —
# that have no typed analogue, so text runs load the text policy instead
# (the typed value is authoritative as written). Tools, DB, and tasks are
# channel-invariant; only the policy the agent reads changes.
_POLICY_PATHS = {
    "voice": INTAKE_MAIN_POLICY_PATH,
    "text": INTAKE_TEXT_POLICY_PATH,
}


def get_environment(
    db: Optional[IntakeDB] = None,
    user_db: Optional[IntakeUserDB] = None,
    solo_mode: bool = False,
    channel: str = "voice",
) -> Environment:
    """Build the intake environment. Solo mode is not supported.

    ``channel`` selects the agent policy (see ``_POLICY_PATHS``). Voice is
    the default so untargeted constructions (registry checks, golden-action
    replay — where the policy text is inert) keep the historical policy;
    the runner passes the run's actual channel (``_build_env_kwargs``).
    """
    if solo_mode:
        raise ValueError("Solo mode not supported for intake domain")
    if channel not in _POLICY_PATHS:
        raise ValueError(
            f"Unknown intake channel {channel!r}: the policy is per channel "
            f"and only knows {sorted(_POLICY_PATHS)}"
        )
    if db is None:
        db = IntakeDB.load(INTAKE_DB_PATH)
    if user_db is None:
        user_db = IntakeUserDB.load(INTAKE_USER_DB_PATH)
    policy = load_file(_POLICY_PATHS[channel])
    return Environment(
        domain_name="intake",
        policy=policy,
        tools=IntakeTools(db),
        user_tools=IntakeUserTools(user_db),
    )


# Flat composition bands (written by ``tau2 intake-tasks compose``): loadable
# as split names, so a run points at a band without touching the canonical
# tasks.json (`tau2 run --domain intake --task-split-name compose_n2`). The
# staged chain bands load through the intake_staged domain instead
# (tau2.domains.intake.staged).
COMPOSE_BAND_SPLITS = ("compose_n2", "compose_n3")


def _load_band_tasks(band: str) -> list[Task]:
    path = INTAKE_DATA_DIR / "bands" / band / "tasks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Compose band {band!r} not found at {path} — run "
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
    """Load the intake tasks. Malformed task files fail loud — no silent skips."""
    if task_split_name in COMPOSE_BAND_SPLITS:
        return _load_band_tasks(task_split_name)
    raw = load_file(INTAKE_TASK_SET_PATH)
    if isinstance(raw, dict) and "tasks" in raw:
        raw = raw["tasks"]
    if not isinstance(raw, list):
        raise ValueError(
            f"Malformed task file {INTAKE_TASK_SET_PATH}: expected a list of tasks "
            f"(or a dict with a 'tasks' list), got {type(raw).__name__}"
        )
    tasks: list[Task] = []
    for index, item in enumerate(raw):
        try:
            tasks.append(Task.model_validate(item))
        except ValidationError as e:
            raise ValueError(
                f"Malformed task at index {index} in {INTAKE_TASK_SET_PATH}: {e}"
            ) from e
    task_ids = [task.id for task in tasks]
    duplicates = sorted(
        {task_id for task_id in task_ids if task_ids.count(task_id) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Duplicate task id(s) in {INTAKE_TASK_SET_PATH}: {', '.join(duplicates)}"
        )
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {sorted(task_splits.keys())}"
        )
    split_ids = task_splits[task_split_name]
    unknown_ids = sorted(set(split_ids) - set(task_ids))
    if unknown_ids:
        raise ValueError(
            f"Split '{task_split_name}' references unknown task id(s): {', '.join(unknown_ids)}"
        )
    return [task for task in tasks if task.id in split_ids]


def get_tasks_split() -> dict[str, list[str]]:
    """Load the task splits. The file must exist and contain a 'base' split."""
    split_path = INTAKE_DATA_DIR / f"split_{INTAKE_TASK_SET_PATH.stem}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Task split file not found: {split_path}")
    task_splits = load_file(split_path)
    if not isinstance(task_splits, dict) or "base" not in task_splits:
        raise ValueError(f"Task split file {split_path} must contain a 'base' split")
    return task_splits


if __name__ == "__main__":
    env = get_environment()
    print(env.get_tools_description("assistant"))
    print(env.get_tools_description("user"))
    print(f"{len(get_tasks())} tasks in the base split")
