from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.hospitality.data_model import HospitalityDB
from tau2.domains.hospitality.tools import HospitalityTools
from tau2.domains.hospitality.user_tools import HospitalityUserTools
from tau2.domains.hospitality.utils import (
    HOSPITALITY_DB_PATH,
    HOSPITALITY_POLICY_PATH,
    HOSPITALITY_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[HospitalityDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Hospitality domain does not support solo mode")
    if db is None:
        db = HospitalityDB.load(HOSPITALITY_DB_PATH)
    tools = HospitalityTools(db)
    # User tools operate on the same DB: guest actions (e.g. online check-in)
    # are immediately visible to the agent, and count toward the DB check.
    user_tools = HospitalityUserTools(db)
    with open(HOSPITALITY_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="hospitality",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(HOSPITALITY_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(HOSPITALITY_TASK_SET_PATH).parent
        / f"split_{Path(HOSPITALITY_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
