import json
import random
from argparse import ArgumentParser
from collections import defaultdict

from tau2.data_model.tasks import Task
from tau2.domains.healthcare.tasks.prescription_issues import (
    prescription_refill_selection_sets,
    is_fixed_prescription_refill,
    get_env_assertions_prescription_refill,
)
from tau2.domains.healthcare.tasks.appointment_issues import (
    appointment_scheduling_selection_sets,
)
from tau2.domains.healthcare.tasks.urgent_triage_issues import (
    urgent_triage_selection_sets,
)
from tau2.domains.healthcare.tasks.chronic_monitoring_issues import (
    chronic_monitoring_selection_sets,
    compose_chronic_monitoring_tasks,
)
from tau2.domains.healthcare.tasks.telehealth_issues import (
    telehealth_setup_selection_sets,
)
from tau2.domains.healthcare.tasks.test_results_issues import (
    test_results_selection_sets,
)
from tau2.domains.healthcare.tasks.patient_mistake_issues import (
    patient_mistake_selection_sets,
)
from tau2.domains.healthcare.tasks.critical_triage_issues import (
    critical_triage_selection_sets,
    is_fixed_critical_triage,
    get_env_assertions_critical_triage,
)
from tau2.domains.healthcare.tasks.manager import TaskManager
from tau2.domains.healthcare.tasks.const import (
    TOOL_CALL_GROUNDING,
    TOOL_CALL_INFO_CHECK,
)
from tau2.domains.healthcare.tasks.utils import get_persona_from_task_id
from tau2.domains.healthcare.tasks.evaluation_functions import (
    is_fixed_appointment_scheduling,
    get_env_assertions_appointment_scheduling,
    is_fixed_urgent_triage,
    get_env_assertions_urgent_triage,
    is_fixed_chronic_monitoring,
    get_env_assertions_chronic_monitoring,
    is_fixed_telehealth_setup,
    get_env_assertions_telehealth_setup,
    is_fixed_test_results_access,
    get_env_assertions_test_results_access,
)
from tau2.domains.healthcare.tasks.patient_mistake_issues import (
    is_fixed_patient_mistake,
    get_env_assertions_patient_mistake,
)
from tau2.utils import DATA_DIR
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall


def get_env_assertions(expected_success: bool) -> list[EnvAssertion]:
    """Placeholder for environment assertions."""
    return []


def set_surrounding(env) -> list[EnvFunctionCall]:
    """Set the patient info for task initialization."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_user_info",
            arguments={
                "name": "Sarah Johnson",
                "patient_id": "patient_001",
                "date_of_birth": "1985-03-15",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_user_location",
            arguments={"location": "home"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": False,
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


def is_fixed(env) -> bool:
    """Placeholder: currently always returns True."""
    return True


prescription_refill_task_manager = TaskManager(
    name="prescription_refill",
    purpose="Test prescription refill request handling.",
    task_instructions=f"Follow the agent's instructions. If they ask you to check medication bottles, use the check_medication_bottle tool. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You need to refill your prescription medication.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) is calling to request a prescription refill. Help them check prescription status and process refill if available, or guide them to contact their doctor if no refills remain.",
    selection_sets=prescription_refill_selection_sets,
    get_env_assertions=get_env_assertions_prescription_refill,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_prescription_refill,
    domain="healthcare",
)

appointment_scheduling_task_manager = TaskManager(
    name="appointment_scheduling",
    purpose="Test appointment scheduling with various constraints.",
    task_instructions=f"Follow the agent's guidance. Check your calendar and insurance when asked. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You want to schedule a medical appointment.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) wants to schedule an appointment. Verify insurance, check doctor availability, and book appointment after getting patient confirmation.",
    selection_sets=appointment_scheduling_selection_sets,
    get_env_assertions=get_env_assertions_appointment_scheduling,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_appointment_scheduling,
    domain="healthcare",
)

urgent_triage_task_manager = TaskManager(
    name="urgent_triage",
    purpose="Test urgent care triage with symptom assessment.",
    task_instructions=f"Describe your symptoms when asked. Use check_symptoms and take_temperature tools as directed. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You are not feeling well and need medical attention.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) is experiencing symptoms. Assess urgency based on fever level, pain severity, and breathing difficulty. Book urgent appointment or transfer to nurse as appropriate.",
    selection_sets=urgent_triage_selection_sets,
    get_env_assertions=get_env_assertions_urgent_triage,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_urgent_triage,
    domain="healthcare",
)

chronic_monitoring_task_manager = TaskManager(
    name="chronic_monitoring",
    purpose="Test chronic condition monitoring with home measurements.",
    task_instructions=f"Share your home monitoring readings when asked. Use measure_blood_pressure, measure_blood_glucose, and measure_oxygen_saturation tools. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You want to discuss your home health monitoring readings.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}. You manage chronic health conditions.",
    ticket="Patient {name} (DOB: {date_of_birth}) is calling about chronic condition monitoring. Review their blood pressure, glucose, and oxygen saturation readings. Schedule follow-up or transfer to nurse for concerning values.",
    selection_sets=chronic_monitoring_selection_sets,
    get_env_assertions=get_env_assertions_chronic_monitoring,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_chronic_monitoring,
    domain="healthcare",
)

telehealth_setup_task_manager = TaskManager(
    name="telehealth_setup",
    purpose="Test telehealth setup with consent and contact management.",
    task_instructions=f"Provide consent when requested. Update emergency contact and acknowledge instructions using the appropriate tools. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You want to set up a telehealth appointment.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) wants to set up telehealth. Obtain necessary consent, verify emergency contact is current, and ensure patient acknowledges any medical instructions.",
    selection_sets=telehealth_setup_selection_sets,
    get_env_assertions=get_env_assertions_telehealth_setup,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_telehealth_setup,
    domain="healthcare",
)

test_results_task_manager = TaskManager(
    name="test_results_access",
    purpose="Test accessing and reviewing medical test results.",
    task_instructions=f"Ask about your test results. Use open_patient_portal to view results when available. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You want to check your medical test results.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) is calling about test results. Check result status, direct to patient portal for ready results, or schedule follow-up for reviewed results. Transfer to nurse for critical findings.",
    selection_sets=test_results_selection_sets,
    get_env_assertions=get_env_assertions_test_results_access,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_test_results_access,
    domain="healthcare",
)

patient_mistake_task_manager = TaskManager(
    name="patient_mistake",
    purpose="Test handling of patient misinformation and verification workflows.",
    task_instructions=f"Report your symptoms or medication concerns as you understand them. The agent may ask you to verify information using tools. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You have concerns about your symptoms or medication.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}.",
    ticket="Patient {name} (DOB: {date_of_birth}) is calling with health concerns. Verify patient-reported information through objective tools (vitals, medication bottles). Use clinical reasoning to identify discrepancies between patient claims and objective data. CRITICAL SAFETY: Life-threatening symptoms (chest pain, difficulty breathing, severe pain) ALWAYS require immediate transfer to clinical staff, regardless of vital sign readings.",
    selection_sets=patient_mistake_selection_sets,
    get_env_assertions=get_env_assertions_patient_mistake,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_patient_mistake,
    domain="healthcare",
)

critical_triage_task_manager = TaskManager(
    name="critical_triage",
    purpose="Test recognition of critical vital signs requiring immediate escalation.",
    task_instructions=f"Report your symptoms as described. You may underestimate the severity. Use home monitoring tools when asked. {TOOL_CALL_INFO_CHECK} {TOOL_CALL_GROUNDING}",
    reason_for_call="You're not feeling well and want to check if you need an appointment.",
    known_info="You are {name}, born on {date_of_birth}, currently at {location}. You have home health monitoring devices.",
    ticket="Patient {name} (DOB: {date_of_birth}) is calling about symptoms. Patient may underestimate severity. CRITICAL: Measure vital signs objectively. Critical values (BP ≥180/120, glucose <70, O2 <85%) require IMMEDIATE transfer to clinical staff, even if patient reports mild symptoms.",
    selection_sets=critical_triage_selection_sets,
    get_env_assertions=get_env_assertions_critical_triage,
    set_surrounding=set_surrounding,
    is_fixed=is_fixed_critical_triage,
    domain="healthcare",
)


def create_tasks(save_tasks: bool = True, max_count_per_bin: int = 3) -> list[Task]:
    tasks: list[Task] = []

    prescription_tasks = prescription_refill_task_manager.create_tasks(save_tasks=False)
    print(f"Number of prescription refill tasks: {len(prescription_tasks)}")
    tasks.extend(prescription_tasks)

    appointment_tasks = appointment_scheduling_task_manager.create_tasks(
        save_tasks=False
    )
    print(f"Number of appointment scheduling tasks: {len(appointment_tasks)}")
    tasks.extend(appointment_tasks)

    urgent_tasks = urgent_triage_task_manager.create_tasks(save_tasks=False)
    print(f"Number of urgent triage tasks: {len(urgent_tasks)}")
    tasks.extend(urgent_tasks)

    # Use custom composition for chronic monitoring to consolidate appointments
    chronic_composed_tasks = compose_chronic_monitoring_tasks()
    chronic_tasks = chronic_monitoring_task_manager.create_tasks(
        save_tasks=False, custom_composed_tasks=chronic_composed_tasks
    )
    print(f"Number of chronic monitoring tasks: {len(chronic_tasks)}")
    tasks.extend(chronic_tasks)

    telehealth_tasks = telehealth_setup_task_manager.create_tasks(save_tasks=False)
    print(f"Number of telehealth setup tasks: {len(telehealth_tasks)}")
    tasks.extend(telehealth_tasks)

    test_results_tasks = test_results_task_manager.create_tasks(save_tasks=False)
    print(f"Number of test results access tasks: {len(test_results_tasks)}")
    tasks.extend(test_results_tasks)

    patient_mistake_tasks = patient_mistake_task_manager.create_tasks(save_tasks=False)
    print(f"Number of patient mistake tasks: {len(patient_mistake_tasks)}")
    tasks.extend(patient_mistake_tasks)

    critical_triage_tasks = critical_triage_task_manager.create_tasks(save_tasks=False)
    print(f"Number of critical triage tasks: {len(critical_triage_tasks)}")
    tasks.extend(critical_triage_tasks)

    print(f"Number of tasks: {len(tasks)}")

    file = DATA_DIR / "tau2" / "domains" / "healthcare" / f"tasks_full.json"
    if save_tasks:
        with open(file, "w") as f:
            json.dump([t.model_dump(exclude_unset=True) for t in tasks], f, indent=2)

    tasks_with_attrs = []
    for intent_tasks, intent in [
        (prescription_tasks, "prescription_refill"),
        (appointment_tasks, "appointment_scheduling"),
        (urgent_tasks, "urgent_triage"),
        (chronic_tasks, "chronic_monitoring"),
        (telehealth_tasks, "telehealth_setup"),
        (test_results_tasks, "test_results_access"),
        (patient_mistake_tasks, "patient_mistake"),
        (critical_triage_tasks, "critical_triage"),
    ]:
        for task in intent_tasks:
            num_subtasks = len(task.id.split("|"))
            tasks_with_attrs.append(
                {
                    "task": task,
                    "intent": intent,
                    "num_subtasks": num_subtasks,
                    "persona": get_persona_from_task_id(task.id),
                }
            )

    file_small = DATA_DIR / "tau2" / "domains" / "healthcare" / f"tasks_small.json"
    small_tasks = [t["task"] for t in tasks_with_attrs if t["num_subtasks"] == 1]
    print(f"Number of tasks in small set: {len(small_tasks)}")
    if save_tasks:
        with open(file_small, "w") as f:
            json.dump(
                [t.model_dump(exclude_unset=True) for t in small_tasks], f, indent=2
            )

    file_sampled = DATA_DIR / "tau2" / "domains" / "healthcare" / f"tasks.json"

    tasks_by_bins = defaultdict(list)
    for task in tasks_with_attrs:
        # Keep tasks with 2+ subtasks, except critical_triage (important despite 1 subtask)
        if task["num_subtasks"] < 2 and task["intent"] != "critical_triage":
            continue
        tasks_by_bins[(task["intent"], task["num_subtasks"], task["persona"])].append(
            task["task"]
        )

    sampled_tasks = []
    for (intent, num_subtasks, persona), tasks_in_bin in tasks_by_bins.items():
        num_sampled = min(max_count_per_bin, len(tasks_in_bin))
        sampled_tasks.extend(random.sample(tasks_in_bin, num_sampled))
        print(
            f"Sampled {num_sampled} tasks for {intent} with {num_subtasks} subtasks and persona {persona}..."
        )

    action_counts = [
        len(task.evaluation_criteria.actions or []) for task in sampled_tasks
    ]
    simple = sum(1 for c in action_counts if c <= 2)
    medium = sum(1 for c in action_counts if 3 <= c <= 4)
    hard = sum(1 for c in action_counts if c >= 5)

    print(f"\nFinal task distribution:")
    print(f"  Total sampled: {len(sampled_tasks)}")
    print(f"  Natural complexity distribution (0-2 / 3-4 / 5+):")
    print(f"    Simple (0-2): {simple} ({simple / len(sampled_tasks) * 100:.1f}%)")
    print(f"    Medium (3-4): {medium} ({medium / len(sampled_tasks) * 100:.1f}%)")
    print(f"    Hard (5+):    {hard} ({hard / len(sampled_tasks) * 100:.1f}%)")
    if save_tasks:
        with open(file_sampled, "w") as f:
            json.dump(
                [t.model_dump(exclude_unset=True) for t in sampled_tasks], f, indent=2
            )

    return tasks


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-m", "--max-count-per-bin", type=int, default=3)
    args = parser.parse_args()
    random.seed(args.seed)
    create_tasks(max_count_per_bin=args.max_count_per_bin)


if __name__ == "__main__":
    main()
