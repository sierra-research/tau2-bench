"""Environment for the vacation rental domain."""

from tau2.data_model.tasks import Task
from tau2.domains.vacation_rental.data_model import VacationRentalDB
from tau2.domains.vacation_rental.tools import VacationRentalTools
from tau2.domains.vacation_rental.utils import (
    VACATION_RENTAL_DB_PATH,
    VACATION_RENTAL_POLICY_PATH,
    VACATION_RENTAL_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: VacationRentalDB | None = None,
    solo_mode: bool = False,
) -> Environment:
    """Create a vacation rental environment.

    Args:
        db: Optional pre-loaded database. If None, loads from default path.
        solo_mode: Whether to run in solo mode (not supported).

    Returns:
        A configured Environment instance.
    """
    if solo_mode:
        raise ValueError("Vacation rental domain does not support solo mode")
    if db is None:
        db = VacationRentalDB.load(VACATION_RENTAL_DB_PATH)
    tools = VacationRentalTools(db)
    with open(VACATION_RENTAL_POLICY_PATH) as fp:
        policy = fp.read()
    return Environment(
        domain_name="vacation_rental",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: str | None = None) -> list[Task]:
    """Load vacation rental tasks.

    Args:
        task_split_name: Optional task split name. If None, returns all tasks.

    Returns:
        List of Task objects.
    """
    tasks = load_file(VACATION_RENTAL_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    return tasks
