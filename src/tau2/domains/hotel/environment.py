# Copyright Sierra
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.hotel.data_model import HotelDB
from tau2.domains.hotel.tools import HotelTools
from tau2.domains.hotel.user_data_model import HotelUserDB
from tau2.domains.hotel.user_tools import HotelUserTools
from tau2.domains.hotel.utils import (
    HOTEL_DB_PATH,
    HOTEL_POLICY_PATH,
    HOTEL_TASKS_PATH,
    HOTEL_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class HotelEnvironment(Environment):
    """Custom environment for hotel domain with agent and user tools."""

    tools: HotelTools
    user_tools: HotelUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: HotelTools,
        user_tools: HotelUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def sync_tools(self):
        """
        Sync agent-side and user-side data.

        This method ensures consistency between the agent's view and the user's context.
        For example, it validates that the guest_id in the user context exists in the agent DB.
        """
        guest_id = self.user_tools.db.guest_context.guest_id

        # If guest_id is set, validate it exists in agent DB
        if guest_id:
            if guest_id not in self.tools.db.guests:
                raise ValueError(f"Guest {guest_id} not found in hotel database")


def get_environment(
    db: Optional[HotelDB] = None,
    user_db: Optional[HotelUserDB] = None,
    solo_mode: bool = False,
) -> HotelEnvironment:
    """
    Create and return a HotelEnvironment instance.

    Args:
        db: Agent-side hotel database (optional, loads from file if not provided)
        user_db: User-side hotel database (optional, loads from file if not provided)
        solo_mode: Whether to run in solo mode (not supported for hotel domain)

    Returns:
        HotelEnvironment instance with both agent and user tools
    """
    if solo_mode:
        raise ValueError("Hotel domain does not support solo mode")

    # Load agent-side database and tools
    if db is None:
        db = HotelDB.load(HOTEL_DB_PATH)
    tools = HotelTools(db)

    # Load user-side database and tools
    if user_db is None:
        user_db = HotelUserDB.load(HOTEL_USER_DB_PATH)
    user_tools = HotelUserTools(user_db)

    # Load policy
    with open(HOTEL_POLICY_PATH, "r") as fp:
        policy = fp.read()

    # Create and return HotelEnvironment
    return HotelEnvironment(
        domain_name="hotel",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(HOTEL_TASKS_PATH)
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
        Path(HOTEL_TASKS_PATH).parent / f"split_{Path(HOTEL_TASKS_PATH).stem}.json"
    )
    return load_file(split_file)
