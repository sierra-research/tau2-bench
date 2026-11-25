# Copyright Sierra
from pathlib import Path
from typing import Optional, cast

from tau2.data_model.tasks import Task
from tau2.domains.healthcare.data_model import HealthcareDB
from tau2.domains.healthcare.tools import HealthcareTools
from tau2.domains.healthcare.user_data_model import HealthcareUserDB
from tau2.domains.healthcare.user_tools import HealthcareUserTools
from tau2.domains.healthcare.utils import (
    HEALTHCARE_DB_PATH,
    HEALTHCARE_POLICY_PATH,
    HEALTHCARE_TASK_SET_PATH,
    HEALTHCARE_TASK_SET_FULL_PATH,
    HEALTHCARE_TASK_SET_SMALL_PATH,
    HEALTHCARE_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class HealthcareEnvironment(Environment):
    """
    Healthcare environment with bidirectional tool support.
    Syncs agent-side and patient-side information.
    """

    tools: HealthcareTools
    user_tools: HealthcareUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: HealthcareTools,
        user_tools: HealthcareUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def make_tool_call(self, tool_name: str, requestor: str = "assistant", **kwargs):
        """Override to track assistant tool calls in real-time for behavioral assertions."""
        if requestor == "assistant":
            self.tools.db.tool_call_history.append(tool_name)

        return super().make_tool_call(tool_name, requestor, **kwargs)

    def set_state(self, initialization_data, initialization_actions, message_history):
        """Override to track tool calls for behavioral assertions."""
        super().set_state(initialization_data, initialization_actions, message_history)

        from tau2.data_model.message import AssistantMessage, UserMessage

        tool_calls = []
        for message in message_history:
            if (
                isinstance(message, (AssistantMessage, UserMessage))
                and message.is_tool_call()
            ):
                for tc in message.tool_calls:
                    if tc.requestor == "assistant":
                        tool_calls.append(tc.name)

        self.tools.db.tool_call_history = tool_calls

    def sync_tools(self):
        """Sync agent and patient tool state."""
        patient_id = self.user_tools.surroundings.patient_id

        if patient_id not in self.tools.db.patients:
            return

        patient = self.tools.db.patients[patient_id]

        if self.user_tools.device.portal_info:
            portal = self.user_tools.device.portal_info

            upcoming_apts = []
            for apt_id in patient.appointment_ids:
                if apt_id in self.tools.db.appointments:
                    apt = self.tools.db.appointments[apt_id]
                    if apt.status == "scheduled":
                        upcoming_apts.append(
                            f"{apt.date} at {apt.time} - {apt.appointment_type} with Dr. {self.tools.db.doctors[apt.doctor_id].name.last_name}"
                        )

            portal.upcoming_appointments = upcoming_apts[:3]

            # outstanding_balance is reserved for a future billing model;
            # the current Payment model tracks completed transactions only
            portal.outstanding_balance = 0


def get_environment(
    db: Optional[HealthcareDB] = None,
    user_db: Optional[HealthcareUserDB] = None,
    solo_mode: bool = False,
) -> HealthcareEnvironment:
    """
    Create a healthcare environment instance.

    Args:
        db: Optional agent-side database. If None, loads from default path.
        user_db: Optional user-side database. If None, loads from default path.
        solo_mode: Whether to run in solo mode (no user interaction)

    Returns:
        Configured HealthcareEnvironment instance
    """
    if db is None:
        db = cast(HealthcareDB, HealthcareDB.load(str(HEALTHCARE_DB_PATH)))

    tools = HealthcareTools(db)

    if not solo_mode:
        if user_db is None:
            user_db = cast(
                HealthcareUserDB, HealthcareUserDB.load(str(HEALTHCARE_USER_DB_PATH))
            )
        user_tools = HealthcareUserTools(user_db)
    else:
        raise ValueError("Healthcare domain does not yet support solo mode")

    with open(HEALTHCARE_POLICY_PATH, "r") as fp:
        policy = fp.read()

    env = HealthcareEnvironment(
        domain_name="healthcare",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )

    return env


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Load healthcare tasks from the task file.

    Args:
        task_split_name: Optional task split name. Supported splits: "base", "train", "test".

    Returns:
        List of Task objects filtered by the specified split
    """
    tasks = load_file(HEALTHCARE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]

    if task_split_name is None:
        return tasks

    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {list(task_splits.keys())}"
        )

    tasks = [task for task in tasks if task.id in task_splits[task_split_name]]
    return tasks


def get_tasks_split() -> dict[str, list[str]]:
    """
    Load task split definitions from split_tasks.json.

    Returns:
        Dictionary mapping split names ("train", "test", "base") to lists of task IDs
    """
    split_file = (
        Path(HEALTHCARE_TASK_SET_PATH).parent
        / f"split_{Path(HEALTHCARE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)


def get_tasks_full(task_split_name: Optional[str] = None) -> list[Task]:
    """Load the full healthcare task set from tasks_full.json."""
    tasks = load_file(HEALTHCARE_TASK_SET_FULL_PATH)
    return [Task.model_validate(task) for task in tasks]


def get_tasks_small(task_split_name: Optional[str] = None) -> list[Task]:
    """Load the small healthcare task set from tasks_small.json."""
    tasks = load_file(HEALTHCARE_TASK_SET_SMALL_PATH)
    return [Task.model_validate(task) for task in tasks]
