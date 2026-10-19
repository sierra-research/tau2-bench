from functools import partial
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.loan_servicing.data_model import LoanServicingDB
from tau2.domains.loan_servicing.tools import LoanServicingTools
from tau2.domains.loan_servicing.utils import (
    LOAN_SERVICING_DB_PATH,
    LOAN_SERVICING_POLICY_FR_PATH,
    LOAN_SERVICING_POLICY_PATH,
    LOAN_SERVICING_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

# Policy language -> (registered domain name, policy file). Both domains share
# the same tools, database, and tasks; only the agent's policy differs.
POLICY_VARIANTS = {
    "en": ("loan_servicing", LOAN_SERVICING_POLICY_PATH),
    "fr": ("loan_servicing_fr", LOAN_SERVICING_POLICY_FR_PATH),
}


def get_environment(
    db: Optional[LoanServicingDB] = None,
    solo_mode: bool = False,
    policy_language: str = "en",
) -> Environment:
    if solo_mode:
        raise ValueError("Loan servicing domain does not support solo mode")
    if policy_language not in POLICY_VARIANTS:
        raise ValueError(f"Invalid policy language: {policy_language}")
    domain_name, policy_path = POLICY_VARIANTS[policy_language]
    if db is None:
        db = LoanServicingDB.load(LOAN_SERVICING_DB_PATH)
    tools = LoanServicingTools(db)
    # The policy contains accented characters, so read it as UTF-8 explicitly.
    policy = load_file(policy_path, encoding="utf-8")
    return Environment(
        domain_name=domain_name,
        policy=policy,
        tools=tools,
    )


get_environment_en_policy = partial(get_environment, policy_language="en")
get_environment_fr_policy = partial(get_environment, policy_language="fr")


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(LOAN_SERVICING_TASK_SET_PATH)
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
        Path(LOAN_SERVICING_TASK_SET_PATH).parent
        / f"split_{Path(LOAN_SERVICING_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
