from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### SelectionSet 1: refills_status_issues


def init_no_refills_remaining(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Set prescription to have 0 refills remaining."""
    rx = env.tools.db.prescriptions["rx_001"]
    doctor = env.tools.db.doctors[rx.doctor_id]

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_refills",
            arguments={"prescription_id": "rx_001", "refills": 0},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_medication_at_home",
            arguments={
                "prescription_number": "rx_001",
                "medication_name": rx.medication_name,
                "dosage": rx.dosage,
                "refills_remaining": 0,
                "prescribing_doctor": f"Dr. {doctor.name.first_name} {doctor.name.last_name}",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_refills_remaining",
            arguments={"prescription_id": "rx_001", "expected_count": 0},
            assert_value=True,
        ),
    ]


def init_has_refills_available(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Set prescription to have refills available."""
    rx = env.tools.db.prescriptions["rx_001"]
    doctor = env.tools.db.doctors[rx.doctor_id]

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_refills",
            arguments={"prescription_id": "rx_001", "refills": 3},
        ),
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
                "medication_name": rx.medication_name,
                "dosage": rx.dosage,
                "refills_remaining": 3,
                "prescribing_doctor": f"Dr. {doctor.name.first_name} {doctor.name.last_name}",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_refills_remaining",
            arguments={"prescription_id": "rx_001", "expected_count": 3},
            assert_value=True,
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_status",
            arguments={"prescription_id": "rx_001", "expected_status": "active"},
            assert_value=True,
        ),
    ]


def fix_has_refills_available(env: HealthcareEnvironment) -> list[ToolCall]:
    """Process refill for prescription with refills available."""
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
            name="verify_insurance_coverage",
            arguments={
                "patient_id": "patient_001",
                "procedure_type": "prescription_refill",
            },
            compare_args=["patient_id"],
        ),
        ToolCall(
            requestor="assistant",
            name="request_prescription_refill",
            arguments={"patient_id": "patient_001", "prescription_id": "rx_001"},
        ),
    ]


def fix_prescription_needs_renewal(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify prescription details then transfer to nurse for renewal."""
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


### SelectionSet 2: prescription_status_issues


def init_prescription_active(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Set prescription status to active with refills available."""
    rx = env.tools.db.prescriptions["rx_001"]
    doctor = env.tools.db.doctors[rx.doctor_id]

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_status",
            arguments={"prescription_id": "rx_001", "status": "active"},
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_refills",
            arguments={"prescription_id": "rx_001", "refills": 3},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_medication_at_home",
            arguments={
                "prescription_number": "rx_001",
                "medication_name": rx.medication_name,
                "dosage": rx.dosage,
                "refills_remaining": 3,
                "prescribing_doctor": f"Dr. {doctor.name.first_name} {doctor.name.last_name}",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_status",
            arguments={"prescription_id": "rx_001", "expected_status": "active"},
            assert_value=True,
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_refills_remaining",
            arguments={"prescription_id": "rx_001", "expected_count": 3},
            assert_value=True,
        ),
    ]


def init_prescription_expired(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Set prescription status to expired."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_status",
            arguments={"prescription_id": "rx_001", "status": "expired"},
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_status",
            arguments={"prescription_id": "rx_001", "expected_status": "expired"},
            assert_value=True,
        ),
    ]


def init_prescription_discontinued(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Set prescription status to discontinued."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_status",
            arguments={"prescription_id": "rx_001", "status": "discontinued"},
        ),
        EnvAssertion(
            env_type="assistant",
            func_name="assert_prescription_status",
            arguments={"prescription_id": "rx_001", "expected_status": "discontinued"},
            assert_value=True,
        ),
    ]


### SelectionSet 3: prescription_type_issues


def init_regular_medication(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Initialize with regular non-controlled medication with refills."""
    rx = env.tools.db.prescriptions["rx_001"]
    doctor = env.tools.db.doctors[rx.doctor_id]

    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_refills",
            arguments={"prescription_id": "rx_001", "refills": 3},
        ),
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
                "medication_name": rx.medication_name,
                "dosage": rx.dosage,
                "refills_remaining": 3,
                "prescribing_doctor": f"Dr. {doctor.name.first_name} {doctor.name.last_name}",
                "pharmacy_name": "Community Pharmacy",
                "pharmacy_phone": "(555) 123-4567",
            },
        ),
    ]


def init_controlled_substance(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Initialize with controlled substance."""
    return [
        EnvFunctionCall(
            env_type="assistant",
            func_name="set_prescription_medication",
            arguments={
                "prescription_id": "rx_001",
                "medication_name": "Oxycodone",
                "dosage": "5mg every 6 hours as needed",
            },
        )
    ]


### Base Tasks

no_refills_remaining_task = BaseTask(
    name="no_refills_remaining",
    description="Patient has 0 refills left on prescription",
    init_funcs=[init_no_refills_remaining],
    fix_funcs=[fix_prescription_needs_renewal],
)

has_refills_available_task = BaseTask(
    name="has_refills_available",
    description="Patient has refills available",
    init_funcs=[init_has_refills_available],
    fix_funcs=[fix_has_refills_available],
)

prescription_active_task = BaseTask(
    name="prescription_active",
    description="Prescription is active",
    init_funcs=[init_prescription_active],
    fix_funcs=[fix_has_refills_available],
)

prescription_expired_task = BaseTask(
    name="prescription_expired",
    description="Prescription has expired",
    init_funcs=[init_prescription_expired],
    fix_funcs=[fix_prescription_needs_renewal],
)

prescription_discontinued_task = BaseTask(
    name="prescription_discontinued",
    description="Prescription was discontinued",
    init_funcs=[init_prescription_discontinued],
    fix_funcs=[fix_prescription_needs_renewal],
)

regular_medication_task = BaseTask(
    name="regular_medication",
    description="Regular non-controlled medication",
    init_funcs=[init_regular_medication],
    fix_funcs=[fix_has_refills_available],
)

controlled_substance_task = BaseTask(
    name="controlled_substance",
    description="Controlled substance",
    init_funcs=[init_controlled_substance],
    fix_funcs=[fix_prescription_needs_renewal],
)


### SelectionSets

refills_status_issues = SelectionSet(
    tasks=[
        no_refills_remaining_task,
        has_refills_available_task,
    ]
)

prescription_status_issues = SelectionSet(
    tasks=[
        prescription_active_task,
        prescription_expired_task,
        prescription_discontinued_task,
    ]
)

prescription_type_issues = SelectionSet(
    tasks=[
        regular_medication_task,
        controlled_substance_task,
    ]
)

prescription_refill_selection_sets = [
    refills_status_issues,
]


### Evaluation Functions


def is_fixed_prescription_refill(env: HealthcareEnvironment) -> bool:
    """Check if the prescription refill issue is resolved."""
    if "rx_001" not in env.tools.db.prescriptions:
        return True

    rx = env.tools.db.prescriptions["rx_001"]

    is_default_state = (
        rx.refills_remaining == 2
        and rx.status == "active"
        and rx.medication_name == "Lisinopril"
    )

    return is_default_state


def get_env_assertions_prescription_refill(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for prescription_refill tasks.

    Args:
        expected_success: Whether the task is expected to be successfully resolved

    Returns:
        Tuple of (env_assertions, nl_assertions, communicate_info)
    """
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_prescription_status",
                arguments={"prescription_id": "rx_001", "expected_status": "active"},
                assert_value=True,
                message="Prescription status should be active for successful refill",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_prescription_refills_remaining",
                arguments={"prescription_id": "rx_001", "expected_count": 2},
                assert_value=True,
                message="Prescription should have 2 refills remaining after processing one refill",
            ),
        ]

        nl_assertions = []

        return (env_assertions, nl_assertions, [])
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "get_prescription_details"},
                assert_value=True,
                message="Agent transferred without checking prescription details first",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_not_called",
                arguments={"tool_name": "request_prescription_refill"},
                assert_value=True,
                message="Agent incorrectly attempted to refill prescription that requires nurse escalation",
            ),
        ]

        nl_assertions = []

        return (env_assertions, nl_assertions, [])
