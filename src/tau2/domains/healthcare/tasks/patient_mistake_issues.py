from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet

### SelectionSet 11: symptom_misreporting_issues


def init_chest_pain_vs_panic_attack(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Patient reports chest pain. Agent must verify vitals and escalate to nurse."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 98.4},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Severe chest pain radiating to left arm, feels like heart attack, racing heart",
                "severity": "severe",
                "duration": "20 minutes",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_symptom_verification",
                "patient_id": "patient_001",
                "reason": "Pending triage - chest pain requiring verification",
            },
        ),
    ]


def init_high_fever_vs_normal_temp(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Patient reports fever but temperature is normal."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_temperature",
            arguments={"temperature": 98.6},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Feeling feverish, very hot and sweaty",
                "severity": "mild",
                "duration": "4 hours",
            },
        ),
    ]


def fix_verify_chest_pain_mismatch(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient identity, check symptoms and vitals, transfer to nurse."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_symptoms", arguments={}),
        ToolCall(requestor="user", name="take_temperature", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def fix_verify_fever_mismatch(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient identity, check symptoms and temperature."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_symptoms", arguments={}),
        ToolCall(requestor="user", name="take_temperature", arguments={}),
    ]


### SelectionSet 12: medication_confusion_issues


def init_wrong_medication_not_working(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Patient confuses medication names. Claims wrong medication isn't working for blood pressure."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_status",
            arguments={"prescription_id": "rx_001", "status": "active"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_medication_at_home",
            arguments={
                "prescription_number": "rx_001",
                "medication_name": "Lisinopril",
                "dosage": "10mg daily",
                "refills_remaining": 2,
                "prescribing_doctor": "Dr. Sarah Martinez",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_medication_at_home",
            arguments={
                "prescription_number": "rx_003",
                "medication_name": "Atorvastatin",
                "dosage": "20mg at bedtime",
                "refills_remaining": 3,
                "prescribing_doctor": "Dr. Sarah Martinez",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_symptom_verification",
                "patient_id": "patient_001",
                "reason": "Pending medication confusion verification",
            },
        ),
    ]


def init_wrong_dosage_confusion(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Patient reports taking incorrect dosage."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_status",
            arguments={"prescription_id": "rx_001", "status": "active"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_medication_at_home",
            arguments={
                "prescription_number": "rx_001",
                "medication_name": "Lisinopril",
                "dosage": "10mg once daily",
                "refills_remaining": 2,
                "prescribing_doctor": "Dr. Sarah Martinez",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_symptom_verification",
                "patient_id": "patient_001",
                "reason": "Pending medication dosage verification",
            },
        ),
    ]


def fix_verify_medication_confusion(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient identity, check medication bottles, look up prescriptions."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_medication_bottle", arguments={}),
        ToolCall(
            requestor="assistant",
            name="get_prescription_details",
            arguments={"prescription_id": "rx_001"},
        ),
        ToolCall(
            requestor="assistant",
            name="get_prescription_details",
            arguments={"prescription_id": "rx_003"},
        ),
    ]


def fix_verify_dosage_confusion(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient identity, check medication bottle, verify prescription, transfer to nurse."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="check_medication_bottle", arguments={}),
        ToolCall(
            requestor="assistant",
            name="get_prescription_details",
            arguments={"prescription_id": "rx_001"},
        ),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


### Base Tasks for SelectionSet 11: symptom_misreporting_issues

chest_pain_vs_panic_attack_task = BaseTask(
    name="chest_pain_vs_panic_attack",
    description="Patient reports severe chest pain; agent must verify vitals and escalate appropriately",
    init_funcs=[init_chest_pain_vs_panic_attack],
    fix_funcs=[fix_verify_chest_pain_mismatch],
)

high_fever_vs_normal_temp_task = BaseTask(
    name="high_fever_vs_normal_temp",
    description="Patient reports high fever, but temperature is normal",
    init_funcs=[init_high_fever_vs_normal_temp],
    fix_funcs=[],
)

### Base Tasks for SelectionSet 12: medication_confusion_issues

wrong_medication_not_working_task = BaseTask(
    name="wrong_medication_not_working",
    description="Patient claims wrong medication isn't working for their condition",
    init_funcs=[init_wrong_medication_not_working],
    fix_funcs=[None],
)

wrong_dosage_confusion_task = BaseTask(
    name="wrong_dosage_confusion",
    description="Patient reports taking incorrect dosage",
    init_funcs=[init_wrong_dosage_confusion],
    fix_funcs=[None],
)

### SelectionSets

symptom_misreporting_issues = SelectionSet(
    tasks=[
        chest_pain_vs_panic_attack_task,
        high_fever_vs_normal_temp_task,
    ]
)

medication_confusion_issues = SelectionSet(
    tasks=[
        wrong_medication_not_working_task,
        wrong_dosage_confusion_task,
    ]
)

patient_mistake_selection_sets = [
    symptom_misreporting_issues,
    medication_confusion_issues,
]


### Evaluation Functions


def is_fixed_patient_mistake(env: HealthcareEnvironment) -> bool:
    """Check if patient mistake scenario is resolved."""
    symptom_marker = env.tools.db.appointments.get("pending_symptom_verification")
    if symptom_marker is not None:
        return False

    if len(env.user_tools.db.patient_device.medications_at_home) >= 2:
        return False

    return True


def get_env_assertions_patient_mistake(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for patient_mistake tasks."""
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "get_patient_details"},
                assert_value=True,
                message="Agent must verify patient identity before addressing medical concerns",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "take_temperature"},
                assert_value=True,
                message="Agent must verify patient's fever claim with objective temperature reading",
            ),
        ]
        nl_assertions = []
        communicate_info = []
        return (env_assertions, nl_assertions, communicate_info)
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "get_patient_details"},
                assert_value=True,
                message="Agent must verify patient identity before addressing medical concerns",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "check_medication_bottle"},
                assert_value=True,
                message="Agent must verify medication details before escalating confusion",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "transfer_to_nurse"},
                assert_value=True,
                message="Agent must escalate patient mistake to nurse for clinical review",
            ),
        ]
        nl_assertions = []
        return (env_assertions, nl_assertions, [])
