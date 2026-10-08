from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### Init Functions


def init_test_results_ready(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Test results are ready and available."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "ready"
        env.tools.db.test_results[
            "test_001"
        ].result = "Complete Blood Count (CBC) - All values within normal range"
        env.tools.db.test_results[
            "test_001"
        ].notes = "Results reviewed and appear normal"
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": True,
                "test_results": [
                    {
                        "test_name": "HbA1c (Diabetes screening)",
                        "test_date": "2024-05-10",
                        "result": "Complete Blood Count (CBC) - All values within normal range",
                        "notes": "Results reviewed and appear normal",
                    }
                ],
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        )
    ]


def init_test_results_pending(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Test results are still pending from lab."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "pending"
        env.tools.db.test_results["test_001"].result = None
        env.tools.db.test_results["test_001"].notes = None
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_test_results_waiting",  # For pending results (non-fixable)
                "patient_id": "patient_001",
                "reason": "Pending test results - awaiting lab",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": False,
                "test_results": [],  # No results yet - still pending
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


def init_test_results_reviewed_by_doctor(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Test results ready but awaiting doctor review before release."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "reviewed"
        env.tools.db.test_results[
            "test_001"
        ].result = "Lipid Panel - Elevated cholesterol levels noted"
        env.tools.db.test_results[
            "test_001"
        ].notes = "Doctor review required before release - follow-up needed"
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_test_results_doctor_review",  # For reviewed results (non-fixable)
                "patient_id": "patient_001",
                "reason": "Pending test results - awaiting doctor review",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": False,  # Not released to patient yet
                "test_results": [],  # Not released to patient - awaiting doctor review
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


### Fix Functions


def fix_provide_test_results_with_followup(
    env: HealthcareEnvironment,
) -> list[ToolCall]:
    """Provide ready test results to patient and schedule routine follow-up."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_test_results",
            arguments={"patient_id": "patient_001", "test_id": "test_001"},
        ),
        ToolCall(
            requestor="assistant",
            name="verify_insurance_coverage",
            arguments={"patient_id": "patient_001", "procedure_type": "follow_up"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
            compare_args=["doctor_id"],
        ),
        ToolCall(requestor="user", name="check_calendar", arguments={}),
        ToolCall(
            requestor="assistant",
            name="book_appointment",
            arguments={
                "patient_id": "patient_001",
                "doctor_id": "doc_001",
                "appointment_type": "follow_up",
                "date": "2024-05-20",
                "time": "14:00",
                "reason": "Routine follow-up to review test results and annual wellness check",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
        ),
    ]


def init_results_normal(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Test results are normal."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "ready"
        env.tools.db.test_results[
            "test_001"
        ].result = "All test values within normal reference ranges"
        env.tools.db.test_results["test_001"].notes = "No abnormalities detected"
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_test_results_routine_followup",
                "patient_id": "patient_001",
                "reason": "Pending routine follow-up for normal test results",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": True,
                "test_results": [
                    {
                        "test_name": "HbA1c (Diabetes screening)",
                        "test_date": "2024-05-10",
                        "result": "All test values within normal reference ranges",
                        "notes": "No abnormalities detected",
                    }
                ],
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


def init_results_abnormal_minor(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Test results show minor abnormalities requiring follow-up."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "ready"
        env.tools.db.test_results[
            "test_001"
        ].result = "Slightly elevated cholesterol (220 mg/dL) - recommend dietary modifications"
        env.tools.db.test_results[
            "test_001"
        ].notes = "Minor abnormality - follow-up in 3 months recommended"

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_test_results_followup",  # For scheduling follow-up
                "patient_id": "patient_001",
                "reason": "Pending test results follow-up - minor abnormalities",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": True,  # Results are ready and available to patient
                "test_results": [
                    {
                        "test_name": "HbA1c (Diabetes screening)",
                        "test_date": "2024-05-10",
                        "result": "Slightly elevated cholesterol (220 mg/dL) - recommend dietary modifications",
                        "notes": "Minor abnormality - follow-up in 3 months recommended",
                    }
                ],
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


def init_results_abnormal_critical(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Test results show critical abnormalities requiring immediate action."""
    if "test_001" in env.tools.db.test_results:
        env.tools.db.test_results["test_001"].status = "reviewed"
        env.tools.db.test_results[
            "test_001"
        ].result = "CRITICAL: Severely elevated glucose (450 mg/dL) and abnormal kidney function"
        env.tools.db.test_results[
            "test_001"
        ].notes = "URGENT - Patient requires immediate medical evaluation"

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_test_results_critical_transfer",  # For emergency transfer
                "patient_id": "patient_001",
                "reason": "Pending test results - critical findings requiring immediate attention",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="set_portal_info",
            arguments={
                "upcoming_appointments": [],
                "recent_visits": [],
                "test_results_available": False,  # Critical results not released to patient, needs nurse review
                "test_results": [],  # Critical results not released - requires nurse review
                "messages_count": 0,
                "outstanding_balance": 0,
            },
        ),
    ]


def fix_schedule_followup_minor(env: HealthcareEnvironment) -> list[ToolCall]:
    """Schedule follow-up appointment for minor abnormalities."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_test_results",
            arguments={"patient_id": "patient_001", "test_id": "test_001"},
        ),
        ToolCall(
            requestor="assistant",
            name="verify_insurance_coverage",
            arguments={"patient_id": "patient_001", "procedure_type": "follow_up"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-30"},
            compare_args=["doctor_id"],
        ),
        ToolCall(requestor="user", name="check_calendar", arguments={}),
        ToolCall(
            requestor="assistant",
            name="book_appointment",
            arguments={
                "patient_id": "patient_001",
                "doctor_id": "doc_001",
                "appointment_type": "follow_up",
                "date": "2024-05-30",
                "time": "10:00",
                "reason": "Follow-up for abnormal test results - discuss findings and treatment plan",
            },
        ),
    ]


def fix_escalate_critical_results(env: HealthcareEnvironment) -> list[ToolCall]:
    """Escalate critical test results to clinical staff."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_test_results",
            arguments={"patient_id": "patient_001", "test_id": "test_001"},
        ),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def fix_inform_test_status(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient and check test results status."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(
            requestor="assistant",
            name="check_test_results",
            arguments={"patient_id": "patient_001", "test_id": "test_001"},
        ),
    ]


### Base Tasks

test_results_ready_task = BaseTask(
    name="ready",
    description="Test results ready and available",
    init_funcs=[init_test_results_ready],
    fix_funcs=[fix_inform_test_status],
)

test_results_pending_task = BaseTask(
    name="pending",
    description="Test results still pending from lab",
    init_funcs=[init_test_results_pending],
    fix_funcs=[fix_inform_test_status],
)

test_results_reviewed_by_doctor_task = BaseTask(
    name="reviewed_by_doctor",
    description="Results awaiting doctor review before release",
    init_funcs=[init_test_results_reviewed_by_doctor],
    fix_funcs=[fix_inform_test_status],
)

results_normal_task = BaseTask(
    name="normal",
    description="Test results normal",
    init_funcs=[init_results_normal],
    fix_funcs=[fix_provide_test_results_with_followup],
)

results_abnormal_minor_task = BaseTask(
    name="abnormal_minor",
    description="Minor abnormalities requiring follow-up",
    init_funcs=[init_results_abnormal_minor],
    fix_funcs=[fix_schedule_followup_minor],
)

results_abnormal_critical_task = BaseTask(
    name="abnormal_critical",
    description="Critical abnormalities requiring immediate action",
    init_funcs=[init_results_abnormal_critical],
    fix_funcs=[fix_escalate_critical_results],
)


### SelectionSets

abnormal_results_issues = SelectionSet(
    tasks=[
        results_normal_task,
        results_abnormal_minor_task,
        results_abnormal_critical_task,
    ]
)

test_results_selection_sets = [
    abnormal_results_issues,
]
