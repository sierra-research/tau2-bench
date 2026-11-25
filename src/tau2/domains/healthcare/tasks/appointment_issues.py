from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### Init Functions


def init_doctor_available(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Doctor has multiple time slots available."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - doctor available",
            },
        )
    ]


def init_limited_availability(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Doctor has limited availability."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - limited availability",
            },
        )
    ]


def init_no_availability_preferred_times(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Doctor has no availability during patient's preferred times."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - preferred times unavailable",
            },
        )
    ]


### Fix Functions


def fix_book_available_appointment(env: HealthcareEnvironment) -> list[ToolCall]:
    """Book appointment in available slot."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(
            requestor="assistant",
            name="verify_insurance_coverage",
            arguments={
                "patient_id": "patient_001",
                "procedure_type": "routine_checkup",
            },
            compare_args=["patient_id"],
        ),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
            compare_args=["doctor_id"],
            accepted_values={"doctor_id": ["doc_001", "doc_003", "doc_006", "doc_008"]},
        ),
        ToolCall(requestor="user", name="check_calendar", arguments={}),
        ToolCall(
            requestor="assistant",
            name="book_appointment",
            arguments={
                "patient_id": "patient_001",
                "doctor_id": "doc_001",
                "appointment_type": "routine_checkup",
                "date": "2024-05-20",
                "time": "14:00",
                "reason": "Routine checkup appointment",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": ["doc_001", "doc_003", "doc_006", "doc_008"]},
        ),
    ]


def init_insurance_verified(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Insurance is on file and verified."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - insurance verified",
            },
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_patient_has_insurance",
            arguments={"patient_id": "patient_001", "expected": True},
            assert_value=True,
        ),
    ]


def init_insurance_not_on_file(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Insurance information is missing from patient record."""
    if "patient_001" in env.tools.db.patients:
        patient = env.tools.db.patients["patient_001"]
        from tau2.domains.healthcare.data_model import InsurancePlan

        patient.insurance = InsurancePlan(
            provider="SelfPay",
            policy_number="",
            group_number="",
            copay_amount=0,
            coverage_details="No insurance on file",
        )
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_insurance_not_on_file",
                "patient_id": "patient_001",
                "reason": "Pending booking request - insurance not on file",
            },
        )
    ]


def init_insurance_coverage_limited(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Insurance has limited coverage for requested service."""
    if "patient_001" in env.tools.db.patients:
        patient = env.tools.db.patients["patient_001"]
        patient.insurance.coverage_details = (
            "Limited coverage - specialist visits require referral"
        )
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_insurance_coverage_limited",
                "patient_id": "patient_001",
                "reason": "Pending booking request - limited insurance coverage",
            },
        )
    ]


def init_no_calendar_conflicts(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Patient has no calendar conflicts."""
    env.user_tools.device.calendar_availability = []
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - no calendar conflicts",
            },
        )
    ]


def init_has_calendar_conflicts(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Patient has conflicts on some proposed dates."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_calendar_slot",
            arguments={
                "date": "2024-05-20",
                "time": "10:00",
                "available": False,
                "reason": "Work meeting",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_calendar_slot",
            arguments={
                "date": "2024-05-20",
                "time": "14:00",
                "available": True,
                "reason": None,
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - has calendar conflicts",
            },
        ),
    ]


def init_routine_checkup(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Simple routine checkup appointment."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_book_appointment",
                "patient_id": "patient_001",
                "reason": "Pending booking request - routine checkup",
            },
        )
    ]


def init_specialist_referral_needed(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Appointment requires specialist referral."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Persistent heart palpitations requiring cardiology evaluation",
                "severity": "moderate",
                "duration": "2 weeks",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_specialist_referral_needed",
                "patient_id": "patient_001",
                "reason": "Pending booking request - specialist referral needed",
            },
        ),
    ]


def init_urgent_care_needed(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Urgent care appointment needed due to severity."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Severe abdominal pain",
                "severity": "severe",
                "duration": "6 hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_urgent_care_needed",
                "patient_id": "patient_001",
                "reason": "Pending booking request - urgent care needed",
            },
        ),
    ]


def fix_urgent_care_appointment(env: HealthcareEnvironment) -> list[ToolCall]:
    """Book urgent care appointment."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
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
            compare_args=["doctor_id"],
            accepted_values={"doctor_id": ["doc_001", "doc_003", "doc_006", "doc_008"]},
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
                "time": "15:00",
                "reason": "Urgent care - severe symptoms requiring immediate evaluation",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": ["doc_001", "doc_003", "doc_006", "doc_008"]},
        ),
    ]


### Base Tasks

doctor_available_task = BaseTask(
    name="doctor_available",
    description="Doctor has multiple available time slots",
    init_funcs=[init_doctor_available],
    fix_funcs=[fix_book_available_appointment],
)

limited_availability_task = BaseTask(
    name="limited_availability",
    description="Doctor has limited availability",
    init_funcs=[init_limited_availability],
    fix_funcs=[fix_book_available_appointment],
)

no_availability_preferred_times_task = BaseTask(
    name="no_availability_preferred_times",
    description="No availability during patient's preferred times",
    init_funcs=[init_no_availability_preferred_times],
    fix_funcs=[fix_book_available_appointment],
)

insurance_verified_task = BaseTask(
    name="insurance_verified",
    description="Insurance on file and verified",
    init_funcs=[init_insurance_verified],
    fix_funcs=[fix_book_available_appointment],
)

insurance_not_on_file_task = BaseTask(
    name="insurance_not_on_file",
    description="Insurance information missing from record",
    init_funcs=[init_insurance_not_on_file],
    fix_funcs=[None],
)

insurance_coverage_limited_task = BaseTask(
    name="insurance_coverage_limited",
    description="Insurance has limited coverage for service",
    init_funcs=[init_insurance_coverage_limited],
    fix_funcs=[None],
)

no_calendar_conflicts_task = BaseTask(
    name="no_calendar_conflicts",
    description="No calendar conflicts",
    init_funcs=[init_no_calendar_conflicts],
    fix_funcs=[fix_book_available_appointment],
)

has_calendar_conflicts_task = BaseTask(
    name="has_calendar_conflicts",
    description="Patient has calendar conflicts on some dates",
    init_funcs=[init_has_calendar_conflicts],
    fix_funcs=[fix_book_available_appointment],
)

routine_checkup_task = BaseTask(
    name="routine_checkup",
    description="Simple routine checkup",
    init_funcs=[init_routine_checkup],
    fix_funcs=[fix_book_available_appointment],
)

specialist_referral_needed_task = BaseTask(
    name="specialist_referral_needed",
    description="Requires specialist referral",
    init_funcs=[init_specialist_referral_needed],
    fix_funcs=[None],
)

urgent_care_needed_task = BaseTask(
    name="urgent_care_needed",
    description="Urgent care appointment needed",
    init_funcs=[init_urgent_care_needed],
    fix_funcs=[fix_urgent_care_appointment],
)


### SelectionSets

doctor_availability_issues = SelectionSet(
    tasks=[
        doctor_available_task,
        limited_availability_task,
        no_availability_preferred_times_task,
    ]
)

insurance_verification_issues = SelectionSet(
    tasks=[
        insurance_verified_task,
    ]
)

calendar_conflict_issues = SelectionSet(
    tasks=[
        no_calendar_conflicts_task,
        has_calendar_conflicts_task,
    ]
)

appointment_type_complexity = SelectionSet(
    tasks=[
        routine_checkup_task,
    ]
)

appointment_scheduling_selection_sets = [
    doctor_availability_issues,
    insurance_verification_issues,
    calendar_conflict_issues,
    appointment_type_complexity,
]
