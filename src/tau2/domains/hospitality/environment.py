"""Environment for the hospitality domain."""

from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.hospitality.data_model import HospitalityDB
from tau2.domains.hospitality.tools import HospitalityTools
from tau2.domains.hospitality.user_data_model import HospitalityUserDB
from tau2.domains.hospitality.user_tools import HospitalityUserTools
from tau2.domains.hospitality.utils import (
    HOSPITALITY_DB_PATH,
    HOSPITALITY_POLICY_PATH,
    HOSPITALITY_TASK_SET_PATH,
    HOSPITALITY_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class HospitalityEnvironment(Environment):
    """Environment for the hospitality domain."""

    tools: HospitalityTools
    user_tools: HospitalityUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: HospitalityTools,
        user_tools: HospitalityUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def sync_tools(self):
        """
        Sync the agent tools with user context.
        This ensures consistency between agent and user state.
        """
        # Sync customer information if available
        if self.user_tools.db.context.phone:
            phone = self.user_tools.db.context.phone
            # Try to find matching customer in agent DB
            for customer in self.tools.db.customers:
                if customer.phone == phone:
                    # Update user context with customer info
                    self.user_tools.db.context.membership_tier = customer.tier.value
                    self.user_tools.db.context.points_balance = customer.points
                    self.user_tools.db.context.previous_visit_count = (
                        customer.visit_count
                    )
                    break


def get_environment(
    db: Optional[HospitalityDB] = None,
    user_db: Optional[HospitalityUserDB] = None,
    solo_mode: bool = False,
) -> HospitalityEnvironment:
    """
    Get an instance of the hospitality environment.

    Args:
        db: Optional pre-loaded database. If None, loads from default path.
        user_db: Optional pre-loaded user database. If None, loads from default path.
        solo_mode: If True, agent has access to both user and assistant tools.

    Returns:
        Configured HospitalityEnvironment instance.
    """
    if db is None:
        db = HospitalityDB.load(HOSPITALITY_DB_PATH)
    tools = HospitalityTools(db)

    if user_db is None:
        if HOSPITALITY_USER_DB_PATH.exists():
            user_db = HospitalityUserDB.load(HOSPITALITY_USER_DB_PATH)
        else:
            user_db = HospitalityUserDB()
    user_tools = HospitalityUserTools(user_db)

    # Load policy
    with open(HOSPITALITY_POLICY_PATH, "r") as fp:
        policy = fp.read()

    env = HospitalityEnvironment(
        domain_name="hospitality",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )

    if solo_mode:
        env.set_solo_mode(True)

    return env


def load_tasks(path: str) -> list[Task]:
    """Load tasks from a data file."""
    tasks = load_file(path)
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    return [Task.model_validate(task) for task in tasks]


def load_tasks_split(path: str) -> Optional[dict[str, list[str]]]:
    """Load tasks split from a data file."""
    split_file = Path(path).parent / f"split_{Path(path).stem}.json"
    if split_file.exists():
        return load_file(split_file)
    return None


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Get tasks for the hospitality domain.

    Args:
        task_split_name: Name of the task split to use. If None, returns all tasks.

    Returns:
        List of Task objects.
    """
    tasks = load_tasks(HOSPITALITY_TASK_SET_PATH)

    if task_split_name is None:
        return tasks

    task_splits = get_tasks_split()
    if task_splits is None:
        return tasks

    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(task_splits.keys())}"
        )

    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> Optional[dict[str, list[str]]]:
    """Get the task splits for the hospitality domain."""
    return load_tasks_split(HOSPITALITY_TASK_SET_PATH)


if __name__ == "__main__":
    env = get_environment()
    print(f"Domain: {env.get_domain_name()}")
    print(f"Number of tools: {len(env.get_tools())}")
    print(f"Number of user tools: {len(env.get_user_tools())}")
    print("\nAgent Tools:")
    for tool in env.get_tools():
        print(f"  - {tool.name}")
    print("\nUser Tools:")
    for tool in env.get_user_tools():
        print(f"  - {tool.name}")
