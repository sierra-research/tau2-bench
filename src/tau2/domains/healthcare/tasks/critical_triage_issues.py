from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### Critical Blood Pressure


def init_hypertensive_crisis_standalone(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Patient reports mild headache, but BP is critically high (hypertensive crisis)."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={
                "has_monitor": True,
                "systolic": 192,
                "diastolic": 128,
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Mild headache, feeling a bit off",
                "severity": "mild",
                "duration": "2 hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_bp_crisis_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - patient reports headache, needs BP check",
            },
        ),
    ]


def fix_hypertensive_crisis_standalone(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient, measure BP, recognize crisis level, transfer immediately."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_pressure", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


### Critical Hypoglycemia


def init_hypoglycemia_standalone(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Patient reports feeling dizzy/shaky, but glucose is dangerously low."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_glucose_monitor",
            arguments={
                "has_monitor": True,
                "glucose_reading": 52,  # <70 is hypoglycemia, <54 is critical
                "measurement_time": "fasting",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Feeling dizzy and shaky, a bit confused",
                "severity": "moderate",
                "duration": "30 minutes",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_glucose_crisis_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - diabetic patient with dizziness",
            },
        ),
    ]


def fix_hypoglycemia_standalone(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient, measure glucose, recognize critical hypoglycemia, transfer."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_glucose", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


### Severe Hypoxemia


def init_severe_hypoxemia_standalone(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Patient reports shortness of breath, oxygen saturation is dangerously low."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_pulse_oximeter",
            arguments={
                "has_monitor": True,
                "spo2": 82,  # <90 is low, <85 is severe
                "heart_rate": 92,  # Normal range for context
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Short of breath when walking, tired",
                "severity": "moderate",
                "duration": "few days",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_oxygen_crisis_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - patient reports breathing difficulty",
            },
        ),
    ]


def fix_severe_hypoxemia_standalone(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient, measure O2, recognize severe hypoxemia, transfer."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_oxygen_saturation", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


### Base Tasks

hypertensive_crisis_standalone_task = BaseTask(
    name="hypertensive_crisis_standalone",
    description="Patient reports mild headache; measured BP shows hypertensive crisis (192/128)",
    init_funcs=[init_hypertensive_crisis_standalone],
    fix_funcs=[fix_hypertensive_crisis_standalone],
)

hypoglycemia_standalone_task = BaseTask(
    name="hypoglycemia_standalone",
    description="Patient reports dizziness; measured glucose shows critical hypoglycemia (52 mg/dL)",
    init_funcs=[init_hypoglycemia_standalone],
    fix_funcs=[fix_hypoglycemia_standalone],
)

severe_hypoxemia_standalone_task = BaseTask(
    name="severe_hypoxemia_standalone",
    description="Patient reports shortness of breath; measured O2 shows severe hypoxemia (82%)",
    init_funcs=[init_severe_hypoxemia_standalone],
    fix_funcs=[fix_severe_hypoxemia_standalone],
)

### Selection Set - These are standalone, not composed

critical_triage_standalone_tasks = SelectionSet(
    tasks=[
        hypertensive_crisis_standalone_task,
        hypoglycemia_standalone_task,
        severe_hypoxemia_standalone_task,
    ]
)

critical_triage_selection_sets = [
    critical_triage_standalone_tasks,
]


### Evaluation Functions


def is_fixed_critical_triage(env: HealthcareEnvironment) -> bool:
    """
    Critical triage tasks check if the transfer was handled.

    These are expected failure scenarios - the task is only "fixed" if
    the agent recognized the critical values and transferred to nurse.

    Since we can't directly check if transfer happened from the environment,
    we check if the appointment marker was resolved (removed).
    """
    markers = [
        appt
        for appt in env.tools.db.appointments.values()
        if appt.appointment_id.startswith("pending_")
    ]
    return len(markers) == 0


def get_env_assertions_critical_triage(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """
    Get environment assertions for critical_triage tasks.

    These tasks test multi-step clinical decision-making:
    1. Verify patient identity
    2. Measure vital signs objectively
    3. Recognize critical values and transfer

    Unlike patient_mistake tasks (which are expected failures with just transfer),
    these tasks expect the agent to complete ALL 3 actions successfully.
    """
    env_assertions = [
        EnvAssertion(
            env_type="assistant",
            func_name="assert_tool_was_called",
            arguments={"tool_name": "transfer_to_nurse"},
            assert_value=True,
            message="Agent must transfer to nurse for critical vital signs",
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_tool_was_not_called",
            arguments={"tool_name": "book_appointment"},
            assert_value=True,
            message="Agent incorrectly tried to book appointment for critical emergency",
        ),
    ]

    nl_assertions = []
    communicate_info = []

    return (env_assertions, nl_assertions, communicate_info)
