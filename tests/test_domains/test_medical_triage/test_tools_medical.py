"""Tests for the medical triage domain tools."""

import pytest

from tau2.domains.medical_triage.data_model import (
    Appointment,
    Insurance,
    MedicalDB,
    Patient,
    Provider,
    Referral,
)
from tau2.domains.medical_triage.tools import MedicalTriageTools


@pytest.fixture
def sample_db() -> MedicalDB:
    """Create a minimal in-memory MedicalDB for testing."""
    return MedicalDB(
        patients={
            "P001": Patient(
                patient_id="P001",
                first_name="Alice",
                last_name="Smith",
                date_of_birth="1985-06-15",
                age=40,
                gender="female",
                phone="555-0101",
                email="alice@example.com",
                allergies=["penicillin", "latex"],
                current_medications=[
                    {"name": "lisinopril", "dosage": "10mg", "frequency": "daily"}
                ],
                medical_history=[
                    {"condition": "hypertension", "date": "2020-01-01", "status": "active"}
                ],
                insurance=Insurance(type="ppo", plan_id="PPO-123", referral_required=False),
                appointments=["APT00001"],
            ),
            "P002": Patient(
                patient_id="P002",
                first_name="Bob",
                last_name="Jones",
                date_of_birth="1970-03-20",
                age=56,
                gender="male",
                phone="555-0202",
                email="bob@example.com",
                allergies=[],
                current_medications=[],
                medical_history=[],
                insurance=Insurance(type="hmo", plan_id="HMO-456", referral_required=True),
                appointments=[],
            ),
            "P003": Patient(
                patient_id="P003",
                first_name="Carol",
                last_name="Davis",
                date_of_birth="1990-11-08",
                age=35,
                gender="female",
                phone="555-0303",
                email="carol@example.com",
                allergies=["aspirin"],
                current_medications=[],
                medical_history=[],
                insurance=Insurance(type="medicare", plan_id="MC-789", referral_required=False),
                appointments=["APT00002", "APT00003", "APT00004"],
            ),
        },
        providers={
            "DR001": Provider(
                provider_id="DR001",
                name="Dr. Wilson",
                department="general_medicine",
                specialty="general_medicine",
                clinic="Main Clinic",
                available_slots=[
                    "2026-03-16T09:00:00",
                    "2026-03-16T10:00:00",
                    "2026-03-17T09:00:00",
                ],
            ),
            "DR002": Provider(
                provider_id="DR002",
                name="Dr. Chen",
                department="cardiology",
                specialty="cardiology",
                clinic="Heart Center",
                available_slots=[
                    "2026-03-17T14:00:00",
                    "2026-03-18T10:00:00",
                ],
            ),
        },
        appointments={
            "APT00001": Appointment(
                appointment_id="APT00001",
                patient_id="P001",
                provider_id="DR001",
                datetime="2026-03-20T09:00:00",
                type="primary_care",
                reason="Annual checkup",
                status="scheduled",
            ),
            "APT00002": Appointment(
                appointment_id="APT00002",
                patient_id="P003",
                provider_id="DR001",
                datetime="2026-03-21T09:00:00",
                type="primary_care",
                reason="Follow-up",
                status="scheduled",
            ),
            "APT00003": Appointment(
                appointment_id="APT00003",
                patient_id="P003",
                provider_id="DR002",
                datetime="2026-03-22T10:00:00",
                type="specialist",
                reason="Cardiology consult",
                status="scheduled",
            ),
            "APT00004": Appointment(
                appointment_id="APT00004",
                patient_id="P003",
                provider_id="DR001",
                datetime="2026-03-23T09:00:00",
                type="primary_care",
                reason="Lab review",
                status="scheduled",
            ),
        },
        referrals={
            "REF00001": Referral(
                referral_id="REF00001",
                patient_id="P002",
                referring_provider_id="DR001",
                target_department="cardiology",
                reason="Chest pain evaluation",
                status="active",
                created_date="2026-03-10",
                expiry_date="2026-06-10",
                used=False,
            ),
        },
    )


@pytest.fixture
def tools(sample_db: MedicalDB) -> MedicalTriageTools:
    return MedicalTriageTools(sample_db)


# ── Patient Lookup ───────────────────────────────────────────────────


class TestFindPatient:
    def test_find_by_name_dob(self, tools: MedicalTriageTools):
        result = tools.find_patient_by_name_dob("Alice", "Smith", "1985-06-15")
        assert result["patient_id"] == "P001"

    def test_find_case_insensitive(self, tools: MedicalTriageTools):
        result = tools.find_patient_by_name_dob("alice", "smith", "1985-06-15")
        assert result["patient_id"] == "P001"

    def test_find_not_found(self, tools: MedicalTriageTools):
        with pytest.raises(ValueError, match="not found"):
            tools.find_patient_by_name_dob("Unknown", "Person", "2000-01-01")


class TestGetPatientDetails:
    def test_get_details(self, tools: MedicalTriageTools):
        result = tools.get_patient_details("P001")
        assert result["first_name"] == "Alice"
        assert result["last_name"] == "Smith"
        assert result["date_of_birth"] == "1985-06-15"

    def test_get_details_not_found(self, tools: MedicalTriageTools):
        with pytest.raises(ValueError, match="not found"):
            tools.get_patient_details("P999")


class TestGetPatientAllergies:
    def test_get_allergies(self, tools: MedicalTriageTools):
        result = tools.get_patient_allergies("P001")
        assert "penicillin" in result["allergies"]
        assert "latex" in result["allergies"]

    def test_get_allergies_empty(self, tools: MedicalTriageTools):
        result = tools.get_patient_allergies("P002")
        assert result["allergies"] == []


class TestGetPatientMedications:
    def test_get_medications(self, tools: MedicalTriageTools):
        result = tools.get_patient_medications("P001")
        assert len(result["medications"]) == 1
        assert result["medications"][0]["name"] == "lisinopril"


class TestGetPatientMedicalHistory:
    def test_get_history(self, tools: MedicalTriageTools):
        result = tools.get_patient_medical_history("P001")
        assert len(result["history"]) == 1
        assert result["history"][0]["condition"] == "hypertension"


class TestGetPatientInsurance:
    def test_get_insurance(self, tools: MedicalTriageTools):
        result = tools.get_patient_insurance("P001")
        assert result["insurance"]["type"] == "ppo"
        assert result["insurance"]["referral_required"] is False

    def test_get_insurance_hmo(self, tools: MedicalTriageTools):
        result = tools.get_patient_insurance("P002")
        assert result["insurance"]["type"] == "hmo"
        assert result["insurance"]["referral_required"] is True


# ── Appointment Read ─────────────────────────────────────────────────


class TestGetAppointmentDetails:
    def test_get_appointment(self, tools: MedicalTriageTools):
        result = tools.get_appointment_details("APT00001")
        assert result["patient_id"] == "P001"
        assert result["status"] == "scheduled"

    def test_get_appointment_not_found(self, tools: MedicalTriageTools):
        with pytest.raises(ValueError, match="not found"):
            tools.get_appointment_details("APT99999")


class TestGetPatientAppointments:
    def test_get_appointments(self, tools: MedicalTriageTools):
        result = tools.get_patient_appointments("P001")
        assert len(result["appointments"]) == 1
        assert result["appointments"][0]["appointment_id"] == "APT00001"


class TestGetAvailableSlots:
    def test_get_slots_by_department(self, tools: MedicalTriageTools):
        result = tools.get_available_slots("general_medicine", "2026-03-16")
        assert len(result["available_slots"]) == 2  # 09:00 and 10:00

    def test_get_slots_date_range(self, tools: MedicalTriageTools):
        result = tools.get_available_slots("general_medicine", "2026-03-16", "2026-03-17")
        assert len(result["available_slots"]) == 3  # 2 on 16th + 1 on 17th

    def test_get_slots_empty(self, tools: MedicalTriageTools):
        result = tools.get_available_slots("dermatology", "2026-03-16")
        assert len(result["available_slots"]) == 0


class TestGetProviderDetails:
    def test_get_provider(self, tools: MedicalTriageTools):
        result = tools.get_provider_details("DR001")
        assert result["name"] == "Dr. Wilson"
        assert result["department"] == "general_medicine"


class TestCheckReferralStatus:
    def test_has_referral(self, tools: MedicalTriageTools):
        result = tools.check_referral_status("P002", "cardiology")
        assert result["has_referral"] is True
        assert result["referral_id"] == "REF00001"

    def test_no_referral(self, tools: MedicalTriageTools):
        result = tools.check_referral_status("P001", "cardiology")
        assert result["has_referral"] is False


# ── Write Tools ──────────────────────────────────────────────────────


class TestScheduleAppointment:
    def test_schedule_basic(self, tools: MedicalTriageTools):
        result = tools.schedule_appointment(
            patient_id="P002",
            appointment_type="primary_care",
            department="general_medicine",
            datetime_slot="2026-03-16T09:00:00",
            provider_id="DR001",
            reason="Checkup",
        )
        assert result["status"] == "scheduled"
        assert result["appointment_id"].startswith("APT")

    def test_schedule_removes_slot(self, tools: MedicalTriageTools):
        tools.schedule_appointment(
            patient_id="P002",
            appointment_type="primary_care",
            department="general_medicine",
            datetime_slot="2026-03-16T09:00:00",
            provider_id="DR001",
            reason="Checkup",
        )
        slots = tools.get_available_slots("general_medicine", "2026-03-16")
        slot_times = [s["datetime"] for s in slots["available_slots"]]
        assert "2026-03-16T09:00:00" not in slot_times

    def test_schedule_max_3_appointments(self, tools: MedicalTriageTools):
        """P003 already has 3 active appointments."""
        with pytest.raises(ValueError, match="3 active appointments"):
            tools.schedule_appointment(
                patient_id="P003",
                appointment_type="primary_care",
                department="general_medicine",
                datetime_slot="2026-03-16T10:00:00",
                provider_id="DR001",
                reason="Extra visit",
            )

    def test_schedule_invalid_slot(self, tools: MedicalTriageTools):
        with pytest.raises(ValueError, match="not available"):
            tools.schedule_appointment(
                patient_id="P002",
                appointment_type="primary_care",
                department="general_medicine",
                datetime_slot="2026-03-16T15:00:00",
                provider_id="DR001",
                reason="Checkup",
            )

    def test_schedule_specialist_needs_referral(self, tools: MedicalTriageTools):
        """P002 has HMO insurance requiring referrals but has one for cardiology."""
        result = tools.schedule_appointment(
            patient_id="P002",
            appointment_type="specialist",
            department="cardiology",
            datetime_slot="2026-03-17T14:00:00",
            provider_id="DR002",
            reason="Chest pain",
        )
        assert result["status"] == "scheduled"

    def test_schedule_specialist_no_referral_fails(self, tools: MedicalTriageTools):
        """P002 has HMO requiring referrals but no referral for orthopedics."""
        with pytest.raises(ValueError, match="referral"):
            tools.schedule_appointment(
                patient_id="P002",
                appointment_type="specialist",
                department="orthopedics",
                datetime_slot="2026-03-17T14:00:00",
                provider_id="DR002",
                reason="Knee pain",
            )


class TestCancelAppointment:
    def test_cancel_scheduled(self, tools: MedicalTriageTools):
        result = tools.cancel_appointment("APT00001")
        assert result["status"] == "cancelled"

    def test_cancel_returns_slot(self, tools: MedicalTriageTools):
        tools.cancel_appointment("APT00001")
        provider = tools.db.providers["DR001"]
        assert "2026-03-20T09:00:00" in provider.available_slots

    def test_cancel_already_cancelled(self, tools: MedicalTriageTools):
        tools.cancel_appointment("APT00001")
        with pytest.raises(ValueError, match="not scheduled"):
            tools.cancel_appointment("APT00001")


class TestRescheduleAppointment:
    def test_reschedule_basic(self, tools: MedicalTriageTools):
        result = tools.reschedule_appointment("APT00001", "2026-03-16T10:00:00")
        assert result["new_datetime"] == "2026-03-16T10:00:00"
        assert result["reschedule_count"] == 1

    def test_reschedule_max_2_times(self, tools: MedicalTriageTools):
        tools.reschedule_appointment("APT00001", "2026-03-16T09:00:00")
        tools.reschedule_appointment("APT00001", "2026-03-16T10:00:00")
        with pytest.raises(ValueError, match="2 times"):
            tools.reschedule_appointment("APT00001", "2026-03-17T09:00:00")

    def test_reschedule_invalid_slot(self, tools: MedicalTriageTools):
        with pytest.raises(ValueError, match="not available"):
            tools.reschedule_appointment("APT00001", "2026-03-16T15:00:00")


class TestCreateReferral:
    def test_create_referral(self, tools: MedicalTriageTools):
        result = tools.create_referral("P001", "DR001", "cardiology", "Heart evaluation")
        assert result["status"] == "active"
        assert result["referral_id"].startswith("REF")


class TestAddTriageNotes:
    def test_add_notes(self, tools: MedicalTriageTools):
        result = tools.add_triage_notes("APT00001", "Patient reports headache", "ESI-4")
        assert result["triage_level"] == "ESI-4"
        assert result["notes_added"] is True


class TestLogEmergencyEscalation:
    def test_log_emergency(self, tools: MedicalTriageTools):
        result = tools.log_emergency_escalation(
            "P001", "Chest pain, shortness of breath", "ESI-1", "Advised 911"
        )
        assert result["escalation_logged"] is True
        assert result["triage_level"] == "ESI-1"


class TestTransferToHumanClinician:
    def test_transfer(self, tools: MedicalTriageTools):
        result = tools.transfer_to_human_clinician("P001", "Patient requests doctor")
        assert result["transferred"] is True


# ── Environment & Registry ───────────────────────────────────────────


class TestEnvironment:
    def test_get_environment(self):
        from tau2.domains.medical_triage.environment import get_environment
        env = get_environment()
        assert env.domain_name == "medical_triage"
        assert env.policy is not None
        assert len(env.policy) > 100

    def test_get_tasks(self):
        from tau2.domains.medical_triage.environment import get_tasks
        tasks = get_tasks()
        assert len(tasks) == 70
        assert all(hasattr(t, "id") for t in tasks)

    def test_get_tasks_split(self):
        from tau2.domains.medical_triage.environment import get_tasks_split
        splits = get_tasks_split()
        assert "base" in splits
        assert len(splits["base"]) > 0

    def test_registry_has_medical(self):
        from tau2.registry import registry
        assert "medical_triage" in registry.get_domains()
        assert "medical_triage" in registry.get_task_sets()

    def test_registry_has_adaptive_agent(self):
        from tau2.registry import registry
        assert "adaptive_agent" in registry.get_agents()

    def test_environment_has_user_tools(self):
        from tau2.domains.medical_triage.environment import get_environment
        env = get_environment()
        assert env.user_tools is not None
        user_tool_names = [t.name for t in env.get_user_tools()]
        assert "check_symptom_severity" in user_tool_names
        assert "take_temperature" in user_tool_names
        assert "check_blood_pressure" in user_tool_names
        assert "check_medication_cabinet" in user_tool_names
        assert "check_insurance_portal" in user_tool_names

    def test_environment_is_dual_control(self):
        from tau2.domains.medical_triage.environment import get_environment, MedicalTriageEnvironment
        env = get_environment()
        assert isinstance(env, MedicalTriageEnvironment)
        # Agent and user tools should be separate
        agent_tools = {t.name for t in env.get_tools()}
        user_tools = {t.name for t in env.get_user_tools()}
        assert len(agent_tools & user_tools) == 0, "Agent and user tools should not overlap"


# ── User Tools ───────────────────────────────────────────────────────


class TestUserTools:
    @pytest.fixture
    def user_tools(self):
        from tau2.domains.medical_triage.user_data_model import (
            MedicalUserDB,
            PatientState,
            PatientDevice,
            HomeMedication,
            InsurancePortalInfo,
        )
        from tau2.domains.medical_triage.user_tools import MedicalUserTools

        user_db = MedicalUserDB(
            patient_state=PatientState(
                patient_id="P001",
                current_symptoms=["headache", "fever"],
                pain_level=6,
                symptom_duration_hours=4.0,
                temperature_f=101.3,
                blood_pressure="140/90",
                pulse=88,
                oxygen_saturation=96,
                home_medications=[
                    HomeMedication(name="ibuprofen", dosage="200mg", quantity_remaining=12, expired=False),
                    HomeMedication(name="amoxicillin", dosage="500mg", quantity_remaining=0, expired=True),
                ],
                insurance_portal=InsurancePortalInfo(
                    plan_type="ppo", copay_amount=30.0,
                    referral_on_file=False, referral_department="",
                    deductible_met=True,
                ),
                devices=PatientDevice(
                    has_thermometer=True,
                    has_bp_monitor=True,
                    has_pulse_oximeter=True,
                    has_glucose_meter=False,
                ),
            )
        )
        return MedicalUserTools(user_db)

    def test_check_symptom_severity(self, user_tools):
        result = user_tools.check_symptom_severity()
        assert result["pain_level"] == 6
        assert "headache" in result["symptoms"]
        assert result["duration_hours"] == 4.0

    def test_take_temperature(self, user_tools):
        result = user_tools.take_temperature()
        assert result["temperature_f"] == 101.3

    def test_take_temperature_no_thermometer(self):
        from tau2.domains.medical_triage.user_data_model import (
            MedicalUserDB, PatientState, PatientDevice,
        )
        from tau2.domains.medical_triage.user_tools import MedicalUserTools
        user_db = MedicalUserDB(
            patient_state=PatientState(
                patient_id="P001",
                devices=PatientDevice(has_thermometer=False),
            )
        )
        tools = MedicalUserTools(user_db)
        result = tools.take_temperature()
        assert result["temperature_f"] is None
        assert "error" in result

    def test_check_blood_pressure(self, user_tools):
        result = user_tools.check_blood_pressure()
        assert result["blood_pressure"] == "140/90"

    def test_check_blood_pressure_no_monitor(self):
        from tau2.domains.medical_triage.user_data_model import (
            MedicalUserDB, PatientState, PatientDevice,
        )
        from tau2.domains.medical_triage.user_tools import MedicalUserTools
        user_db = MedicalUserDB(
            patient_state=PatientState(
                patient_id="P001",
                devices=PatientDevice(has_bp_monitor=False),
            )
        )
        tools = MedicalUserTools(user_db)
        result = tools.check_blood_pressure()
        assert result["blood_pressure"] is None

    def test_check_pulse_oximeter(self, user_tools):
        result = user_tools.check_pulse_oximeter()
        assert result["oxygen_saturation"] == 96
        assert result["pulse"] == 88

    def test_check_medication_cabinet(self, user_tools):
        result = user_tools.check_medication_cabinet()
        meds = result["medications_at_home"]
        assert len(meds) == 2
        assert meds[0]["name"] == "ibuprofen"
        assert meds[1]["expired"] is True

    def test_check_insurance_portal(self, user_tools):
        result = user_tools.check_insurance_portal()
        assert result["plan_type"] == "ppo"
        assert result["copay"] == 30.0
        assert result["referral_on_file"] is False

    def test_vital_readings_accumulate(self, user_tools):
        user_tools.take_temperature()
        user_tools.check_blood_pressure()
        user_tools.check_pulse_oximeter()
        readings = user_tools.db.patient_state.vital_readings
        types = [r.type for r in readings]
        assert "temperature" in types
        assert "blood_pressure" in types
        assert "oxygen" in types


class TestSyncTools:
    def test_sync_referral_to_portal(self):
        """When agent creates a referral, patient's portal should reflect it."""
        from tau2.domains.medical_triage.environment import get_environment
        env = get_environment()
        # Set patient ID in user state
        env.user_tools.set_patient_info("PAT001")
        # Set insurance portal
        from tau2.domains.medical_triage.user_data_model import InsurancePortalInfo
        env.user_tools.db.patient_state.insurance_portal = InsurancePortalInfo(
            plan_type="ppo", copay_amount=30.0,
            referral_on_file=False, referral_department="",
        )
        # Agent creates a referral
        env.tools.create_referral("PAT001", "PRV001", "cardiology", "Chest pain")
        # Sync
        env.sync_tools()
        # Patient portal should now show referral
        portal = env.user_tools.db.patient_state.insurance_portal
        assert portal.referral_on_file is True
        assert portal.referral_department == "cardiology"
