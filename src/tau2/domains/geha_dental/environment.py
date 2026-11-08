from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.geha_dental.data_model import GehaDentalDB
from tau2.domains.geha_dental.tools import GehaDentalTools
from tau2.domains.geha_dental.utils import (
    GEHA_DENTAL_DB_PATH,
    GEHA_DENTAL_POLICY_PATH,
    GEHA_DENTAL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[GehaDentalDB] = None,
    solo_mode: bool = False,
    policy: str | None = None,
) -> Environment:
    if solo_mode:
        raise ValueError("Solo mode not supported for geha_dental")
    if db is None:
        db = GehaDentalDB.load(GEHA_DENTAL_DB_PATH)
    else:
        db = db.model_copy(deep=True)
    tools = GehaDentalTools(db)
    if policy is None:
        with open(GEHA_DENTAL_POLICY_PATH, "r") as fp:
            policy = fp.read()
    return Environment(domain_name="geha_dental", policy=policy, tools=tools)


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = [Task.model_validate(t) for t in load_file(GEHA_DENTAL_TASK_SET_PATH)]
    if task_split_name is None:
        return tasks
    splits = get_tasks_split()
    if task_split_name not in splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {splits.keys()}"
        )
    return [t for t in tasks if t.id in splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    path = Path(GEHA_DENTAL_TASK_SET_PATH)
    return load_file(path.parent / f"split_{path.stem}.json")
