from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet


### Init Functions


def init_consent_not_required(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """No consent required for this interaction."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="provide_consent",
            arguments={"consent_type": "telehealth"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="provide_consent",
            arguments={"consent_type": "data_sharing"},
        ),
    ]


def init_telehealth_consent_needed(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall | EnvAssertion]:
    """Telehealth consent required but not yet provided."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="provide_consent",
            arguments={"consent_type": "data_sharing"},
        )
    ]


def init_data_sharing_consent_needed(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Data sharing consent required for specialist referral."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="provide_consent",
            arguments={"consent_type": "telehealth"},
        )
    ]


### Fix Functions


def fix_obtain_telehealth_consent(env: HealthcareEnvironment) -> list[ToolCall]:
    """Obtain telehealth consent from patient."""
    return [
        ToolCall(
            requestor="user",
            name="provide_consent",
            arguments={"consent_type": "telehealth"},
        )
    ]


def fix_obtain_data_sharing_consent(env: HealthcareEnvironment) -> list[ToolCall]:
    """Obtain data sharing consent from patient."""
    return [
        ToolCall(
            requestor="user",
            name="provide_consent",
            arguments={"consent_type": "data_sharing"},
        )
    ]


def init_emergency_contact_current(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Emergency contact is current and on file."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_emergency_contact",
            arguments={
                "name": "Jane Smith",
                "phone": "555-0102",
                "relationship": "spouse",
            },
        )
    ]


def init_emergency_contact_missing(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """No emergency contact on file."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_emergency_contact",
            arguments={
                "name": "MISSING - No emergency contact on file",
                "phone": "000-0000",
                "relationship": "none",
            },
        )
    ]


def init_emergency_contact_outdated(
    env: HealthcareEnvironment,
) -> list[EnvFunctionCall]:
    """Emergency contact information is outdated."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_emergency_contact",
            arguments={
                "name": "Old Contact (disconnected)",
                "phone": "555-9999",
                "relationship": "friend",
            },
        )
    ]


def fix_update_emergency_contact(env: HealthcareEnvironment) -> list[ToolCall]:
    """Update emergency contact information."""
    return [
        ToolCall(
            requestor="user",
            name="update_emergency_contact",
            arguments={
                "name": "Emergency Contact",
                "phone": "555-0000",
                "relationship": "family",
            },
            compare_args=[],
        )
    ]


def init_no_instructions_needed(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """No special instructions required."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "medication"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "post_care"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "pre_surgery"},
        ),
    ]


def init_medication_instructions(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Medication instructions need acknowledgment."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "post_care"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "pre_surgery"},
        ),
    ]


def init_post_care_instructions(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Post-care instructions need acknowledgment."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "medication"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "pre_surgery"},
        ),
    ]


def init_pre_surgery_instructions(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Pre-surgery instructions need acknowledgment."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "medication"},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="acknowledge_instructions",
            arguments={"instruction_type": "post_care"},
        ),
    ]


def fix_acknowledge_medication(env: HealthcareEnvironment) -> list[ToolCall]:
    """Acknowledge medication instructions."""
    return [
        ToolCall(
            requestor="user",
            name="acknowledge_instructions",
            arguments={"instruction_type": "medication"},
            compare_args=["instruction_type"],
        )
    ]


def fix_acknowledge_post_care(env: HealthcareEnvironment) -> list[ToolCall]:
    """Acknowledge post-care instructions."""
    return [
        ToolCall(
            requestor="user",
            name="acknowledge_instructions",
            arguments={"instruction_type": "post_care"},
            compare_args=["instruction_type"],
        )
    ]


def fix_acknowledge_pre_surgery(env: HealthcareEnvironment) -> list[ToolCall]:
    """Acknowledge pre-surgery instructions."""
    return [
        ToolCall(
            requestor="user",
            name="acknowledge_instructions",
            arguments={"instruction_type": "pre_surgery"},
            compare_args=["instruction_type"],
        )
    ]


### Base Tasks

consent_not_required_task = BaseTask(
    name="not_required",
    description="No consent required",
    init_funcs=[],
    fix_funcs=[],
)

telehealth_consent_needed_task = BaseTask(
    name="telehealth_consent_needed",
    description="Telehealth consent required",
    init_funcs=[init_telehealth_consent_needed],
    fix_funcs=[fix_obtain_telehealth_consent],
)

data_sharing_consent_needed_task = BaseTask(
    name="data_sharing_consent_needed",
    description="Data sharing consent required",
    init_funcs=[init_data_sharing_consent_needed],
    fix_funcs=[fix_obtain_data_sharing_consent],
)

emergency_contact_current_task = BaseTask(
    name="current",
    description="Emergency contact current and on file",
    init_funcs=[],
    fix_funcs=[],
)

emergency_contact_missing_task = BaseTask(
    name="missing",
    description="No emergency contact on file",
    init_funcs=[init_emergency_contact_missing],
    fix_funcs=[fix_update_emergency_contact],
)

emergency_contact_outdated_task = BaseTask(
    name="outdated",
    description="Emergency contact information outdated",
    init_funcs=[init_emergency_contact_outdated],
    fix_funcs=[fix_update_emergency_contact],
)

no_instructions_needed_task = BaseTask(
    name="no_instructions_needed",
    description="No special instructions required",
    init_funcs=[],
    fix_funcs=[],
)

medication_instructions_task = BaseTask(
    name="medication_instructions",
    description="Medication instructions need acknowledgment",
    init_funcs=[init_medication_instructions],
    fix_funcs=[fix_acknowledge_medication],
)

post_care_instructions_task = BaseTask(
    name="post_care_instructions",
    description="Post-care instructions need acknowledgment",
    init_funcs=[init_post_care_instructions],
    fix_funcs=[fix_acknowledge_post_care],
)

pre_surgery_instructions_task = BaseTask(
    name="pre_surgery_instructions",
    description="Pre-surgery instructions need acknowledgment",
    init_funcs=[init_pre_surgery_instructions],
    fix_funcs=[fix_acknowledge_pre_surgery],
)


### SelectionSets

consent_issues = SelectionSet(
    tasks=[
        telehealth_consent_needed_task,
        data_sharing_consent_needed_task,
    ]
)

emergency_contact_issues = SelectionSet(
    tasks=[
        emergency_contact_missing_task,
        emergency_contact_outdated_task,
    ]
)

instruction_acknowledgment_issues = SelectionSet(
    tasks=[
        medication_instructions_task,
        post_care_instructions_task,
        pre_surgery_instructions_task,
    ]
)

telehealth_setup_selection_sets = [
    consent_issues,
    emergency_contact_issues,
    instruction_acknowledgment_issues,
]
