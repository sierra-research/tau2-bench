from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### Init Functions


def init_no_fever(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Temperature normal."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 98.6},
        )
    ]


def init_mild_fever(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Mild fever 99-101°F."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 100.2},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Mild fever",
                "severity": "mild",
                "duration": "1 day",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - mild fever",
            },
        ),
    ]


def init_high_fever(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """High fever 101-103°F."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 102.1},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "High fever with chills",
                "severity": "moderate",
                "duration": "2 days",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - high fever",
            },
        ),
    ]


def init_very_high_fever(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Very high fever >103°F (urgent)."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 103.8},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Very high fever with severe chills",
                "severity": "severe",
                "duration": "6 hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - very high fever",
            },
        ),
    ]


### Fix Functions


def fix_urgent_fever(env: HealthcareEnvironment) -> list[ToolCall]:
    """Book urgent appointment for high fever."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="take_temperature", arguments={}),
        ToolCall(
            requestor="assistant",
            name="verify_insurance_coverage",
            arguments={"patient_id": "patient_001", "procedure_type": "urgent_care"},
            compare_args=["patient_id"],
        ),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
        ),
        ToolCall(requestor="user", name="check_calendar", arguments={}),
        ToolCall(
            requestor="assistant",
            name="book_appointment",
            arguments={
                "patient_id": "patient_001",
                "doctor_id": "doc_001",
                "appointment_type": "urgent_care",
                "date": "2024-05-20",
                "time": "16:00",
                "reason": "High fever requiring urgent evaluation",
            },
        ),
    ]


def init_no_pain(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """No significant pain."""
    return []


def init_mild_pain(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Mild pain 1-3/10."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Mild headache",
                "severity": "mild",
                "duration": "few hours",
            },
        )
    ]


def init_moderate_pain(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Moderate pain 4-6/10."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Moderate body aches and muscle pain",
                "severity": "moderate",
                "duration": "1 day",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_triage",
                "patient_id": "patient_001",
                "reason": "Pending triage - moderate pain",
            },
        ),
    ]


def init_severe_pain(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Severe pain 7-10/10 (urgent)."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Severe chest pain radiating to left arm",
                "severity": "severe",
                "duration": "30 minutes",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_transfer",
                "patient_id": "patient_001",
                "reason": "Pending triage - severe pain requiring transfer",
            },
        ),
    ]


def fix_urgent_pain(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for severe pain triage."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_symptoms", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def fix_very_high_fever(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for very high fever (≥103°F)."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="take_temperature", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def init_normal_breathing(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """No breathing issues."""
    return []


def init_difficulty_breathing(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Shortness of breath."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Difficulty breathing, shortness of breath",
                "severity": "severe",
                "duration": "2 hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_transfer",
                "patient_id": "patient_001",
                "reason": "Pending triage - difficulty breathing requiring transfer",
            },
        ),
    ]


def fix_breathing_emergency(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for breathing difficulty."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_symptoms", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def fix_no_urgent_symptoms(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient and symptoms, confirm no urgent care needed."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_symptoms", arguments={}),
    ]


### Base Tasks

no_fever_task = BaseTask(
    name="no_fever",
    description="Temperature normal",
    init_funcs=[init_no_fever],
    fix_funcs=[fix_no_urgent_symptoms],
)

mild_fever_task = BaseTask(
    name="mild_fever",
    description="Mild fever 99-101°F",
    init_funcs=[init_mild_fever],
    fix_funcs=[fix_urgent_fever],
)

high_fever_task = BaseTask(
    name="high_fever",
    description="High fever 101-103°F",
    init_funcs=[init_high_fever],
    fix_funcs=[fix_urgent_fever],
)

very_high_fever_task = BaseTask(
    name="very_high_fever",
    description="Very high fever >103°F",
    init_funcs=[init_very_high_fever],
    fix_funcs=[fix_very_high_fever],
)

no_pain_task = BaseTask(
    name="no_pain",
    description="No significant pain",
    init_funcs=[init_no_pain],
    fix_funcs=[fix_no_urgent_symptoms],
)

mild_pain_task = BaseTask(
    name="mild_pain",
    description="Mild pain 1-3/10",
    init_funcs=[init_mild_pain],
    fix_funcs=[fix_no_urgent_symptoms],
)

moderate_pain_task = BaseTask(
    name="moderate_pain",
    description="Moderate pain 4-6/10",
    init_funcs=[init_moderate_pain],
    fix_funcs=[fix_urgent_fever],
)

severe_pain_task = BaseTask(
    name="severe_pain",
    description="Severe pain 7-10/10",
    init_funcs=[init_severe_pain],
    fix_funcs=[fix_urgent_pain],
)

normal_breathing_task = BaseTask(
    name="normal_breathing",
    description="No breathing issues",
    init_funcs=[init_normal_breathing],
    fix_funcs=[fix_no_urgent_symptoms],
)

difficulty_breathing_task = BaseTask(
    name="difficulty_breathing",
    description="Shortness of breath",
    init_funcs=[init_difficulty_breathing],
    fix_funcs=[fix_breathing_emergency],
)


### SelectionSets

fever_level_issues = SelectionSet(
    tasks=[
        mild_fever_task,
        high_fever_task,
        very_high_fever_task,
    ]
)

pain_severity_issues = SelectionSet(
    tasks=[
        moderate_pain_task,
    ]
)

breathing_issues = SelectionSet(tasks=[])

urgent_triage_selection_sets = [
    fever_level_issues,
    pain_severity_issues,
    breathing_issues,
]
