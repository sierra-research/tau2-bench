"""Evaluation functions for healthcare domain intents."""

from tau2.data_model.tasks import EnvAssertion
from tau2.domains.healthcare.environment import HealthcareEnvironment


# ============================================================================
# APPOINTMENT_SCHEDULING evaluation functions
# ============================================================================


def get_env_assertions_appointment_scheduling(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for appointment_scheduling tasks."""
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_appointment_count_exceeds_baseline",
                arguments={},
                assert_value=True,
                message="No new appointment was created - appointment count did not exceed baseline",
            )
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent verified patient identity before accessing appointment information",
            # "Agent confirmed insurance coverage and copay amount with patient",
            # "Agent checked doctor availability before suggesting appointment times",
            # "Agent confirmed appointment details (date, time, doctor, reason) with patient before booking",
            # "Agent provided clear appointment confirmation information"
        ]

        return (env_assertions, nl_assertions, [])
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "get_patient_details"},
                assert_value=True,
                message="Agent transferred without verifying patient identity first",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_not_called",
                arguments={"tool_name": "book_appointment"},
                assert_value=True,
                message="Agent incorrectly attempted to book appointment when transfer was required",
            ),
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent explained why appointment cannot be scheduled",
            # "Agent provided alternative options or next steps"
        ]

        return (env_assertions, nl_assertions, [])


def is_fixed_appointment_scheduling(env: HealthcareEnvironment) -> bool:
    """Check if appointment scheduling request is resolved."""
    assertions, _, _ = get_env_assertions_appointment_scheduling(expected_success=True)
    success = True
    for assertion in assertions:
        success = success and env.run_env_assertion(
            assertion,
            raise_assertion_error=False,
        )
    return success


# ============================================================================
# URGENT_TRIAGE evaluation functions
# ============================================================================


def get_env_assertions_urgent_triage(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for urgent_triage tasks."""
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_appointment_count_exceeds_baseline",
                arguments={},
                assert_value=True,
                message="No urgent appointment was created - appointment count did not exceed baseline",
            )
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent assessed all reported symptoms systematically",
            # "Agent checked for red flag symptoms (high fever, severe pain, breathing difficulty)",
            # "Agent asked patient to measure temperature if available",
            # "Agent made appropriate triage decision based on symptom severity",
            # "Agent clearly communicated next steps to patient"
        ]

        return (env_assertions, nl_assertions, [])
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "check_symptoms"},
                assert_value=True,
                message="Agent transferred without assessing symptoms first",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "transfer_to_nurse"},
                assert_value=True,
                message="Agent identified severe symptoms but failed to transfer to clinical team",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_not_called",
                arguments={"tool_name": "book_appointment"},
                assert_value=True,
                message="Agent incorrectly attempted to book appointment when transfer was required",
            ),
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent recognized severe/critical symptoms requiring immediate clinical attention",
            # "Agent explained need for nurse/clinical review",
            # "Agent transferred patient promptly"
        ]

        return (env_assertions, nl_assertions, [])


def is_fixed_urgent_triage(env: HealthcareEnvironment) -> bool:
    """Check if urgent triage request is resolved."""
    assertions, _, _ = get_env_assertions_urgent_triage(expected_success=True)
    success = True
    for assertion in assertions:
        success = success and env.run_env_assertion(
            assertion,
            raise_assertion_error=False,
        )
    return success


# ============================================================================
# CHRONIC_MONITORING evaluation functions
# ============================================================================


def get_env_assertions_chronic_monitoring(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for chronic_monitoring tasks."""
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_appointment_count_exceeds_baseline",
                arguments={},
                assert_value=True,
                message="No monitoring appointment was created - appointment count did not exceed baseline",
            )
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent requested all relevant home monitoring readings (BP, glucose, SpO2 as appropriate)",
            # "Agent asked patient to take measurements if not recently done",
            # "Agent assessed whether readings are within normal ranges for patient's conditions",
            # "Agent made appropriate recommendation based on readings",
            # "Agent provided clear guidance on when to seek further care"
        ]

        return (env_assertions, nl_assertions, [])
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "transfer_to_nurse"},
                assert_value=True,
                message="Agent detected critical vitals but failed to transfer to nurse for emergency care",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_not_called",
                arguments={"tool_name": "book_appointment"},
                assert_value=True,
                message="Agent incorrectly attempted to book appointment when transfer was required",
            ),
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent identified concerning vital sign readings requiring clinical review",
            # "Agent explained need for nurse evaluation",
            # "Agent transferred patient for immediate clinical assessment"
        ]

        return (env_assertions, nl_assertions, [])


def is_fixed_chronic_monitoring(env: HealthcareEnvironment) -> bool:
    """Check if chronic monitoring request is resolved."""
    assertions, _, _ = get_env_assertions_chronic_monitoring(expected_success=True)
    success = True
    for assertion in assertions:
        success = success and env.run_env_assertion(
            assertion,
            raise_assertion_error=False,
        )
    return success


# ============================================================================
# TELEHEALTH_SETUP evaluation functions
# ============================================================================


def is_fixed_telehealth_setup(env: HealthcareEnvironment) -> bool:
    """Check if telehealth setup request is resolved."""
    consents_provided = set(env.user_tools.device.consents_provided or [])
    acknowledged_instructions = set(
        env.user_tools.device.acknowledged_instructions or []
    )
    emergency_contact = env.user_tools.surroundings.emergency_contact

    if emergency_contact is not None:
        contact_str = str(emergency_contact)
        if (
            "Old Contact" in contact_str
            or "disconnected" in contact_str
            or "MISSING" in contact_str
        ):
            return False

    if len(consents_provided) == 0 and len(acknowledged_instructions) == 0:
        return True

    required_consents = {"telehealth", "data_sharing"}
    required_instructions = {"medication", "post_care", "pre_surgery"}

    if len(consents_provided) > 0:
        if not required_consents.issubset(consents_provided):
            return False

    if len(acknowledged_instructions) > 0:
        if not required_instructions.issubset(acknowledged_instructions):
            return False

    return True


def get_env_assertions_telehealth_setup(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for telehealth_setup tasks."""
    if expected_success:
        env_assertions = []

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent obtained all required telehealth consents from patient",
            # "Agent verified emergency contact information is current",
            # "Agent ensured patient acknowledged all medical instructions",
            # "Agent confirmed patient has necessary technology for telehealth session",
            # "Agent provided clear information about telehealth appointment process"
        ]

        return (env_assertions, nl_assertions, [])
    else:
        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent identified missing required consents or information",
            # "Agent explained what is needed to complete telehealth setup"
        ]

        return ([], nl_assertions, [])


# ============================================================================
# TEST_RESULTS_ACCESS evaluation functions
# ============================================================================


def get_env_assertions_test_results_access(
    expected_success: bool,
) -> tuple[list[EnvAssertion], list[str], list[str]]:
    """Get environment assertions for test_results_access tasks."""
    if expected_success:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_appointment_count_exceeds_baseline",
                arguments={},
                assert_value=True,
                message="No follow-up appointment was created - appointment count did not exceed baseline",
            )
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent verified patient identity before discussing test results",
            # "Agent checked test result status (ready, pending, or under review)",
            # "Agent directed patient to appropriate resource (portal for ready results, follow-up for abnormalities)",
            # "Agent explained test result findings in patient-friendly language if available",
            # "Agent scheduled follow-up if abnormalities were detected"
        ]

        return (env_assertions, nl_assertions, [])
    else:
        env_assertions = [
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_called",
                arguments={"tool_name": "check_test_results"},
                assert_value=True,
                message="Agent transferred without checking test results first",
            ),
            EnvAssertion(
                env_type="assistant",
                func_name="assert_tool_was_not_called",
                arguments={"tool_name": "book_appointment"},
                assert_value=True,
                message="Agent incorrectly attempted to book appointment when transfer was required",
            ),
        ]

        # Planned: enable when NL evaluation is supported in the framework
        nl_assertions = [
            # "Agent identified critical/urgent test results requiring clinical review",
            # "Agent did not release critical results directly to patient without clinical review",
            # "Agent transferred patient to nurse for proper clinical evaluation"
        ]

        return (env_assertions, nl_assertions, [])


def is_fixed_test_results_access(env: HealthcareEnvironment) -> bool:
    """Check if test results access request is resolved."""
    assertions, _, _ = get_env_assertions_test_results_access(expected_success=True)
    success = True
    for assertion in assertions:
        success = success and env.run_env_assertion(
            assertion,
            raise_assertion_error=False,
        )
    return success
