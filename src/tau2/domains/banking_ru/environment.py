from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.banking_ru.data_model import BankingDB
from tau2.domains.banking_ru.tools import BankingTools
from tau2.domains.banking_ru.utils import (
    BANKING_RU_DB_PATH,
    BANKING_RU_POLICY_PATH,
    BANKING_RU_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[BankingDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Solo mode not supported for banking_ru")
    if db is None:
        db = BankingDB.load(BANKING_RU_DB_PATH)
    tools = BankingTools(db)
    with open(BANKING_RU_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
    return Environment(
        domain_name="banking_ru",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(BANKING_RU_TASK_SET_PATH)
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
        Path(BANKING_RU_TASK_SET_PATH).parent
        / f"split_{Path(BANKING_RU_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
