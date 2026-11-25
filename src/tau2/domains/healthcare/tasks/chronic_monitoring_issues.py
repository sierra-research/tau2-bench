from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.healthcare.environment import HealthcareEnvironment
from tau2.domains.healthcare.tasks.utils import BaseTask, SelectionSet, ComposedTask


### Init Functions


def init_bp_normal(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood pressure in normal range <120/80."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={"has_monitor": True, "systolic": 118, "diastolic": 78},
        )
    ]


def init_bp_elevated(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood pressure elevated 120-129/<80."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={"has_monitor": True, "systolic": 125, "diastolic": 78},
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_bp_followup",  # Shared for all BP monitoring appointments
                "patient_id": "patient_001",
                "reason": "Pending BP monitoring follow-up",
            },
        ),
    ]


def init_bp_stage1(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood pressure Stage 1 Hypertension 130-139/80-89."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={"has_monitor": True, "systolic": 135, "diastolic": 85},
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_bp_followup",  # Shared for all BP monitoring appointments
                "patient_id": "patient_001",
                "reason": "Pending BP monitoring follow-up - Stage 1 hypertension",
            },
        ),
    ]


def init_bp_stage2(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood pressure Stage 2 Hypertension e140/90."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={"has_monitor": True, "systolic": 148, "diastolic": 94},
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_bp_followup",  # Shared for all BP monitoring appointments
                "patient_id": "patient_001",
                "reason": "Pending BP monitoring follow-up - Stage 2 hypertension",
            },
        ),
    ]


def init_bp_hypertensive_crisis(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood pressure Hypertensive Crisis e180/120."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_bp_monitor",
            arguments={"has_monitor": True, "systolic": 185, "diastolic": 122},
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_chronic_emergency_transfer",  # For emergency transfers
                "patient_id": "patient_001",
                "reason": "Pending emergency transfer - hypertensive crisis",
            },
        ),
    ]


### Fix Functions


_GP_DOCTORS = ["doc_001", "doc_003", "doc_006", "doc_008"]


def fix_bp_elevated_monitor(env: HealthcareEnvironment) -> list[ToolCall]:
    """Schedule follow-up for elevated BP."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_pressure", arguments={}),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
            compare_args=["doctor_id"],
            accepted_values={"doctor_id": _GP_DOCTORS},
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
                "time": "15:00",
                "reason": "Follow-up for elevated blood pressure monitoring",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": _GP_DOCTORS},
        ),
    ]


def fix_bp_crisis_emergency(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for hypertensive crisis."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_pressure", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def fix_bp_normal_routine(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient and blood pressure, inform all is well."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_pressure", arguments={}),
    ]


def fix_glucose_normal_routine(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient and glucose, inform all is well."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_glucose", arguments={}),
    ]


def fix_spo2_normal_routine(env: HealthcareEnvironment) -> list[ToolCall]:
    """Verify patient and oxygen saturation, inform all is well."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_oxygen_saturation", arguments={}),
    ]


def init_glucose_normal(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood glucose normal 70-100 mg/dL fasting."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_glucose_monitor",
            arguments={
                "has_monitor": True,
                "glucose_reading": 92,
                "measurement_time": "fasting",
            },
        )
    ]


def init_glucose_prediabetes(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood glucose prediabetes range 100-125 mg/dL."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_glucose_monitor",
            arguments={
                "has_monitor": True,
                "glucose_reading": 112,
                "measurement_time": "fasting",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_glucose_followup",  # Shared for glucose monitoring
                "patient_id": "patient_001",
                "reason": "Pending glucose monitoring follow-up - prediabetes",
            },
        ),
    ]


def init_glucose_diabetes(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood glucose diabetes range e126 mg/dL fasting."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_glucose_monitor",
            arguments={
                "has_monitor": True,
                "glucose_reading": 145,
                "measurement_time": "fasting",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_glucose_followup",  # Shared for glucose monitoring
                "patient_id": "patient_001",
                "reason": "Pending glucose monitoring follow-up - diabetes",
            },
        ),
    ]


def init_glucose_hypoglycemia(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Blood glucose hypoglycemia <70 mg/dL."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_glucose_monitor",
            arguments={
                "has_monitor": True,
                "glucose_reading": 62,
                "measurement_time": "random",
            },
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Feeling shaky and dizzy",
                "severity": "moderate",
                "duration": "30 minutes",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_chronic_emergency_transfer",  # For emergency transfers
                "patient_id": "patient_001",
                "reason": "Pending emergency transfer - hypoglycemia",
            },
        ),
    ]


def fix_glucose_monitoring(env: HealthcareEnvironment) -> list[ToolCall]:
    """Schedule appointment for glucose monitoring."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_glucose", arguments={}),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
            compare_args=["doctor_id"],
            accepted_values={"doctor_id": _GP_DOCTORS},
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
                "time": "09:00",
                "reason": "Blood glucose monitoring and diabetes management",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": _GP_DOCTORS},
        ),
    ]


def fix_glucose_hypoglycemia_emergency(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for hypoglycemia."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_blood_glucose", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


def init_spo2_normal(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Oxygen saturation normal 95-100%."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_pulse_oximeter",
            arguments={"has_monitor": True, "spo2": 97, "heart_rate": 72},
        )
    ]


def init_spo2_mild_hypoxemia(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Oxygen saturation mild hypoxemia 90-94%."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_pulse_oximeter",
            arguments={"has_monitor": True, "spo2": 92, "heart_rate": 78},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Mild shortness of breath",
                "severity": "mild",
                "duration": "few hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_spo2_followup",  # Shared for oxygen monitoring
                "patient_id": "patient_001",
                "reason": "Pending oxygen saturation monitoring - mild hypoxemia",
            },
        ),
    ]


def init_spo2_moderate_hypoxemia(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Oxygen saturation moderate hypoxemia 85-89%."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_pulse_oximeter",
            arguments={"has_monitor": True, "spo2": 87, "heart_rate": 88},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Moderate shortness of breath, feeling winded",
                "severity": "moderate",
                "duration": "several hours",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_spo2_followup",  # Shared for oxygen monitoring
                "patient_id": "patient_001",
                "reason": "Pending oxygen saturation monitoring - moderate hypoxemia",
            },
        ),
    ]


def init_spo2_severe_hypoxemia(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
    """Oxygen saturation severe hypoxemia <85%."""
    return [
        EnvFunctionCall(
            env_type="user",
            func_name="set_pulse_oximeter",
            arguments={"has_monitor": True, "spo2": 82, "heart_rate": 95},
        ),
        EnvFunctionCall(
            env_type="user",
            func_name="add_symptom",
            arguments={
                "description": "Severe difficulty breathing, gasping for air",
                "severity": "severe",
                "duration": "1 hour",
            },
        ),
        EnvFunctionCall(
            env_type="assistant",
            func_name="create_appointment_marker",
            arguments={
                "appointment_id": "pending_chronic_emergency_transfer",  # For emergency transfers
                "patient_id": "patient_001",
                "reason": "Pending emergency transfer - severe hypoxemia",
            },
        ),
    ]


def fix_spo2_monitoring(env: HealthcareEnvironment) -> list[ToolCall]:
    """Schedule appointment for oxygen monitoring."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_oxygen_saturation", arguments={}),
        ToolCall(
            requestor="assistant",
            name="check_available_time_slots",
            arguments={"doctor_id": "doc_001", "date": "2024-05-20"},
            compare_args=["doctor_id"],
            accepted_values={"doctor_id": _GP_DOCTORS},
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
                "time": "11:00",
                "reason": "Low oxygen saturation requiring urgent evaluation",
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": _GP_DOCTORS},
        ),
    ]


def fix_spo2_emergency(env: HealthcareEnvironment) -> list[ToolCall]:
    """Transfer to nurse for severe hypoxemia."""
    return [
        ToolCall(
            requestor="assistant",
            name="get_patient_details",
            arguments={"full_name": "Sarah Johnson", "date_of_birth": "1985-03-15"},
        ),
        ToolCall(requestor="user", name="measure_oxygen_saturation", arguments={}),
        ToolCall(requestor="assistant", name="transfer_to_nurse", arguments={}),
    ]


### Composition Functions


def consolidate_chronic_monitoring_appointments(
    env: HealthcareEnvironment, fix_funcs: list
) -> list[ToolCall]:
    """
    Consolidate multiple appointment bookings into a single comprehensive appointment.

    When multiple non-critical vitals need monitoring, medical practice dictates ONE
    comprehensive appointment covering all conditions, not separate appointments.

    This function:
    1. Identifies all unique appointment booking actions from fix_funcs
    2. If multiple appointments would be booked, consolidates them into ONE
    3. Combines the appointment reasons into a comprehensive reason
    4. Keeps all other actions (identity verification, vital measurements, etc.)
    """
    all_tool_calls = []
    for func in fix_funcs:
        if func is not None:
            all_tool_calls.extend(func(env))

    deduplicated_calls = []
    seen_identity_verification = False
    seen_check_slots = False
    seen_check_calendar = False
    appointment_calls = []

    for tc in all_tool_calls:
        if tc.name == "get_patient_details" and tc.requestor == "assistant":
            if not seen_identity_verification:
                deduplicated_calls.append(tc)
                seen_identity_verification = True
        elif tc.name == "check_available_time_slots" and tc.requestor == "assistant":
            if not seen_check_slots:
                deduplicated_calls.append(tc)
                seen_check_slots = True
        elif tc.name == "check_calendar" and tc.requestor == "user":
            if not seen_check_calendar:
                deduplicated_calls.append(tc)
                seen_check_calendar = True
        elif tc.name == "book_appointment":
            appointment_calls.append(tc)
        else:
            deduplicated_calls.append(tc)

    if len(appointment_calls) > 1:
        reasons = [call.arguments.get("reason", "") for call in appointment_calls]
        consolidated_reason = "Review home monitoring readings - " + ", ".join(
            set(
                [
                    r.replace(
                        "Follow-up for elevated blood pressure monitoring",
                        "blood pressure",
                    )
                    .replace(
                        "Blood glucose monitoring and diabetes management",
                        "glucose/diabetes",
                    )
                    .replace(
                        "Low oxygen saturation requiring urgent evaluation",
                        "oxygen saturation",
                    )
                    for r in reasons
                    if r
                ]
            )
        )

        consolidated_appointment = ToolCall(
            requestor="assistant",
            name="book_appointment",
            arguments={
                "patient_id": appointment_calls[0].arguments["patient_id"],
                "doctor_id": appointment_calls[0].arguments["doctor_id"],
                "appointment_type": "follow_up",
                "date": appointment_calls[0].arguments["date"],
                "time": appointment_calls[0].arguments["time"],
                "reason": consolidated_reason,
            },
            compare_args=["patient_id", "doctor_id", "appointment_type"],
            accepted_values={"doctor_id": _GP_DOCTORS},
        )
        deduplicated_calls.append(consolidated_appointment)
    elif len(appointment_calls) == 1:
        deduplicated_calls.append(appointment_calls[0])

    return deduplicated_calls


### Base Tasks

bp_normal_task = BaseTask(
    name="BP_normal",
    description="Blood pressure normal <120/80",
    init_funcs=[init_bp_normal],
    fix_funcs=[fix_bp_normal_routine],
)

bp_elevated_task = BaseTask(
    name="elevated",
    description="Blood pressure elevated 120-129/<80",
    init_funcs=[init_bp_elevated],
    fix_funcs=[fix_bp_elevated_monitor],
)

bp_stage1_task = BaseTask(
    name="stage1",
    description="Stage 1 Hypertension 130-139/80-89",
    init_funcs=[init_bp_stage1],
    fix_funcs=[fix_bp_elevated_monitor],
)

bp_stage2_task = BaseTask(
    name="stage2",
    description="Stage 2 Hypertension e140/90",
    init_funcs=[init_bp_stage2],
    fix_funcs=[fix_bp_elevated_monitor],
)

bp_hypertensive_crisis_task = BaseTask(
    name="hypertensive_crisis",
    description="Hypertensive Crisis e180/120",
    init_funcs=[init_bp_hypertensive_crisis],
    fix_funcs=[fix_bp_crisis_emergency],
)

glucose_normal_task = BaseTask(
    name="normal",
    description="Blood glucose normal 70-100 mg/dL",
    init_funcs=[init_glucose_normal],
    fix_funcs=[fix_glucose_normal_routine],
)

glucose_prediabetes_task = BaseTask(
    name="prediabetes",
    description="Prediabetes range 100-125 mg/dL",
    init_funcs=[init_glucose_prediabetes],
    fix_funcs=[fix_glucose_monitoring],
)

glucose_diabetes_task = BaseTask(
    name="diabetes",
    description="Diabetes range e126 mg/dL fasting",
    init_funcs=[init_glucose_diabetes],
    fix_funcs=[fix_glucose_monitoring],
)

glucose_hypoglycemia_task = BaseTask(
    name="hypoglycemia",
    description="Hypoglycemia <70 mg/dL",
    init_funcs=[init_glucose_hypoglycemia],
    fix_funcs=[fix_glucose_hypoglycemia_emergency],
)

spo2_normal_task = BaseTask(
    name="normal",
    description="Oxygen saturation normal 95-100%",
    init_funcs=[init_spo2_normal],
    fix_funcs=[fix_spo2_normal_routine],
)

spo2_mild_hypoxemia_task = BaseTask(
    name="mild_hypoxemia",
    description="Mild hypoxemia 90-94%",
    init_funcs=[init_spo2_mild_hypoxemia],
    fix_funcs=[fix_spo2_monitoring],
)

spo2_moderate_hypoxemia_task = BaseTask(
    name="moderate_hypoxemia",
    description="Moderate hypoxemia 85-89%",
    init_funcs=[init_spo2_moderate_hypoxemia],
    fix_funcs=[fix_spo2_emergency],
)

spo2_severe_hypoxemia_task = BaseTask(
    name="severe_hypoxemia",
    description="Severe hypoxemia <85%",
    init_funcs=[init_spo2_severe_hypoxemia],
    fix_funcs=[fix_spo2_emergency],
)


### SelectionSets

blood_pressure_issues = SelectionSet(
    tasks=[
        bp_elevated_task,
        bp_stage1_task,
        bp_stage2_task,
    ]
)

blood_glucose_issues = SelectionSet(
    tasks=[
        glucose_prediabetes_task,
        glucose_diabetes_task,
    ]
)

oxygen_saturation_issues = SelectionSet(
    tasks=[
        spo2_mild_hypoxemia_task,
    ]
)

chronic_monitoring_selection_sets = [
    blood_pressure_issues,
    blood_glucose_issues,
    oxygen_saturation_issues,
]


### Custom Composition for Chronic Monitoring


def consolidate_init_appointment_markers(
    init_funcs: list,
) -> list:
    """
    Consolidate multiple appointment markers in init functions.

    When multiple appointment markers would be created, consolidate them into one
    generic "pending_chronic_monitoring" marker.
    """
    consolidated_init_funcs = []

    for func in init_funcs:
        consolidated_init_funcs.append(func)

    # Create a wrapper that will consolidate appointment markers
    def create_consolidated_init():
        def consolidated_init(env: HealthcareEnvironment) -> list[EnvFunctionCall]:
            all_calls = []
            appointment_markers = []

            for func in init_funcs:
                calls = func(env)
                for call in calls:
                    if (
                        isinstance(call, EnvFunctionCall)
                        and call.func_name == "create_appointment_marker"
                    ):
                        appointment_markers.append(call)
                    else:
                        all_calls.append(call)

            if len(appointment_markers) > 1:
                all_calls.append(
                    EnvFunctionCall(
                        env_type="assistant",
                        func_name="create_appointment_marker",
                        arguments={
                            "appointment_id": "pending_chronic_monitoring",
                            "patient_id": "patient_001",
                            "reason": "Pending chronic condition monitoring follow-up",
                        },
                    )
                )
            elif len(appointment_markers) == 1:
                all_calls.append(appointment_markers[0])

            return all_calls

        return consolidated_init

    return [create_consolidated_init()]


def compose_chronic_monitoring_tasks() -> list[ComposedTask]:
    """
    Compose chronic monitoring tasks with custom logic to consolidate appointments.

    When multiple non-critical vitals need monitoring, creates tasks that expect ONE
    comprehensive appointment instead of multiple separate appointments.
    """
    from itertools import product

    selection_sets = chronic_monitoring_selection_sets
    product_tasks = list(
        product(*[selection_set.tasks + [None] for selection_set in selection_sets])
    )
    composed_tasks = []

    for tasks in product_tasks:
        tasks = sorted([t for t in tasks if t is not None], key=lambda x: x.name)
        if len(tasks) == 0:
            continue

        init_funcs_raw = [f for t in tasks for f in t.init_funcs]
        has_emergency = any(None in t.fix_funcs for t in tasks)
        non_none_fix_funcs = [f for t in tasks for f in t.fix_funcs if f is not None]

        if has_emergency:
            fix_funcs = [None]
            init_funcs = init_funcs_raw
        elif len(non_none_fix_funcs) > 1:
            init_funcs = consolidate_init_appointment_markers(init_funcs_raw)

            def create_consolidated_fix_func(funcs_to_consolidate):
                def consolidated_fix(env: HealthcareEnvironment) -> list[ToolCall]:
                    return consolidate_chronic_monitoring_appointments(
                        env, funcs_to_consolidate
                    )

                return consolidated_fix

            fix_funcs = [create_consolidated_fix_func(non_none_fix_funcs)]
        elif len(non_none_fix_funcs) == 1:
            fix_funcs = non_none_fix_funcs
            init_funcs = init_funcs_raw
        else:
            fix_funcs = []
            init_funcs = init_funcs_raw

        extra_env_assertions = [f for t in tasks for f in t.extra_env_assertions]

        composed_task = ComposedTask(
            name="|".join([t.name for t in tasks]),
            description=", ".join([t.description for t in tasks]),
            composed_from=tasks,
            init_funcs=init_funcs,
            fix_funcs=fix_funcs,
            extra_env_assertions=extra_env_assertions,
        )
        composed_tasks.append(composed_task)

    return composed_tasks
