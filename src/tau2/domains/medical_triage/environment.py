# Medical triage domain — dual-control environment setup
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.medical_triage.data_model import MedicalDB
from tau2.domains.medical_triage.tools import MedicalTriageTools
from tau2.domains.medical_triage.user_data_model import MedicalUserDB
from tau2.domains.medical_triage.user_tools import MedicalUserTools
from tau2.domains.medical_triage.utils import (
    MEDICAL_DB_PATH,
    MEDICAL_POLICY_PATH,
    MEDICAL_TASK_SET_PATH,
    MEDICAL_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class MedicalTriageEnvironment(Environment):
    """Dual-control medical triage environment.

    The agent has access to medical records, scheduling, and triage tools.
    The patient (user) has access to home medical devices, medication cabinet,
    and insurance portal — creating realistic information asymmetry.

    This mirrors the telecom domain's dual-control pattern where the user
    controls their device while the agent controls the backend systems.
    """

    tools: MedicalTriageTools
    user_tools: MedicalUserTools

    def sync_tools(self) -> None:
        """Synchronize state between agent tools and user tools.

        Called after each tool call to propagate state changes.
        For example, if the agent schedules an appointment, the user
        should be able to see that reflected in their state.
        """
        if self.user_tools is None or self.tools is None:
            return

        patient_id = self.user_tools.db.patient_state.patient_id
        if not patient_id or patient_id not in self.tools.db.patients:
            return

        # Sync insurance referral status: if agent created a referral,
        # patient's portal should reflect it
        if self.user_tools.db.patient_state.insurance_portal is not None:
            portal = self.user_tools.db.patient_state.insurance_portal
            for ref in self.tools.db.referrals.values():
                if ref.patient_id == patient_id and ref.status == "active":
                    portal.referral_on_file = True
                    portal.referral_department = ref.target_department
                    break


def get_environment(
    db: Optional[MedicalDB] = None,
    user_db: Optional[MedicalUserDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if db is None:
        db = MedicalDB.load(MEDICAL_DB_PATH)
    tools = MedicalTriageTools(db)

    user_tools = None
    if MEDICAL_USER_DB_PATH.exists():
        if user_db is None:
            user_db = MedicalUserDB.load(MEDICAL_USER_DB_PATH)
        user_tools = MedicalUserTools(user_db)

    with open(MEDICAL_POLICY_PATH, "r") as fp:
        policy = fp.read()

    if user_tools is not None:
        return MedicalTriageEnvironment(
            domain_name="medical_triage",
            policy=policy,
            tools=tools,
            user_tools=user_tools,
            solo_mode=solo_mode,
        )
    return Environment(
        domain_name="medical_triage",
        policy=policy,
        tools=tools,
        solo_mode=solo_mode,
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
