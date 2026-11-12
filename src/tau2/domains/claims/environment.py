# Copyright Sierra
from pathlib import Path
from typing import Optional
import os
import sys

from tau2.data_model.tasks import Task
from tau2.domains.claims.data_model import InsuranceClaimsDB
from tau2.domains.claims.tools import ClaimsTools
from tau2.domains.claims.utils import (
    CLAIMS_DB_PATH,
    TASK_SET_PATH,
    POLICY_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[InsuranceClaimsDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Claims domain does not support solo mode")
    if db is None:
        db = InsuranceClaimsDB.model_validate_json(open(CLAIMS_DB_PATH, "r", encoding="utf-8").read())
    tools = ClaimsTools(db)
    with open(POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
    return Environment(
        domain_name="claims",
        policy=policy,
        tools=tools,
    )

def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(TASK_SET_PATH).parent
        / f"split_{Path(TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
