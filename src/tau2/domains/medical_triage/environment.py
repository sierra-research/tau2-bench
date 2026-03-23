# Medical triage domain — environment setup
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.medical_triage.data_model import MedicalDB
from tau2.domains.medical_triage.tools import MedicalTriageTools
from tau2.domains.medical_triage.utils import (
    MEDICAL_DB_PATH,
    MEDICAL_POLICY_PATH,
    MEDICAL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[MedicalDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Medical triage domain does not support solo mode")
    if db is None:
        db = MedicalDB.load(MEDICAL_DB_PATH)
    tools = MedicalTriageTools(db)
    with open(MEDICAL_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="medical_triage",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(MEDICAL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(task_splits.keys())}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(MEDICAL_TASK_SET_PATH).parent
        / f"split_{Path(MEDICAL_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
