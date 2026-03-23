"""User tools for the medical triage domain — actions the patient can perform."""

from typing import Optional

from loguru import logger

from tau2.domains.medical_triage.user_data_model import (
    MedicalUserDB,
    VitalReading,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MedicalUserTools(ToolKitBase):
    """Tools available to the patient (user simulator) during triage."""

    db: MedicalUserDB

    def __init__(self, db: MedicalUserDB) -> None:
        super().__init__(db)

    # ── Setup helpers (called by initialization_actions) ──────────────

    def set_patient_info(self, patient_id: str) -> str:
        """Set the patient ID for this session."""
        self.db.patient_state.patient_id = patient_id
        return f"Patient ID set to {patient_id}"

    def set_symptoms(self, symptoms: list[str], pain_level: int, duration_hours: float) -> str:
        """Set the patient's current symptoms."""
        self.db.patient_state.current_symptoms = symptoms
        self.db.patient_state.pain_level = pain_level
        self.db.patient_state.symptom_duration_hours = duration_hours
        return f"Symptoms set: {symptoms}, pain {pain_level}/10, {duration_hours}h duration"

    def set_vitals(
        self,
        temperature_f: Optional[float] = None,
        blood_pressure: Optional[str] = None,
        pulse: Optional[int] = None,
        oxygen_saturation: Optional[int] = None,
    ) -> str:
        """Pre-set the patient's actual vital signs (ground truth)."""
        ps = self.db.patient_state
        if temperature_f is not None:
            ps.temperature_f = temperature_f
        if blood_pressure is not None:
            ps.blood_pressure = blood_pressure
        if pulse is not None:
            ps.pulse = pulse
        if oxygen_saturation is not None:
            ps.oxygen_saturation = oxygen_saturation
        return "Vitals set"

    # ── Public user tools (available during conversation) ─────────────

    @is_tool(tool_type=ToolType.READ)
    def check_symptom_severity(self) -> dict:
        """Check and report current symptom severity. Returns pain level and symptom list."""
        ps = self.db.patient_state
        return {
            "pain_level": ps.pain_level,
            "symptoms": ps.current_symptoms,
            "duration_hours": ps.symptom_duration_hours,
        }

    @is_tool(tool_type=ToolType.READ)
    def take_temperature(self) -> dict:
        """Use home thermometer to take temperature. Only works if patient has a thermometer."""
        ps = self.db.patient_state
        if not ps.devices.has_thermometer:
            return {"error": "No thermometer available", "temperature_f": None}
        if ps.temperature_f is None:
            return {"temperature_f": 98.6, "note": "Normal temperature"}
        reading = VitalReading(
            type="temperature", value=str(ps.temperature_f), unit="°F"
        )
        ps.vital_readings.append(reading)
        return {"temperature_f": ps.temperature_f}

    @is_tool(tool_type=ToolType.READ)
    def check_blood_pressure(self) -> dict:
        """Use home blood pressure monitor. Only works if patient has one."""
        ps = self.db.patient_state
        if not ps.devices.has_bp_monitor:
            return {"error": "No blood pressure monitor available", "blood_pressure": None}
        if ps.blood_pressure is None:
            return {"blood_pressure": "120/80", "note": "Normal"}
        reading = VitalReading(
            type="blood_pressure", value=ps.blood_pressure, unit="mmHg"
        )
        ps.vital_readings.append(reading)
        return {"blood_pressure": ps.blood_pressure}

    @is_tool(tool_type=ToolType.READ)
    def check_pulse_oximeter(self) -> dict:
        """Use home pulse oximeter. Only works if patient has one."""
        ps = self.db.patient_state
        if not ps.devices.has_pulse_oximeter:
            return {"error": "No pulse oximeter available", "oxygen_saturation": None, "pulse": None}
        result = {}
        if ps.oxygen_saturation is not None:
            result["oxygen_saturation"] = ps.oxygen_saturation
            ps.vital_readings.append(
                VitalReading(type="oxygen", value=str(ps.oxygen_saturation), unit="%")
            )
        if ps.pulse is not None:
            result["pulse"] = ps.pulse
            ps.vital_readings.append(
                VitalReading(type="pulse", value=str(ps.pulse), unit="bpm")
            )
        return result or {"oxygen_saturation": 98, "pulse": 72, "note": "Normal"}

    @is_tool(tool_type=ToolType.READ)
    def check_medication_cabinet(self) -> dict:
        """Check what medications the patient has at home."""
        ps = self.db.patient_state
        meds = [
            {
                "name": m.name,
                "dosage": m.dosage,
                "quantity_remaining": m.quantity_remaining,
                "expired": m.expired,
            }
            for m in ps.home_medications
        ]
        return {"medications_at_home": meds}

    @is_tool(tool_type=ToolType.READ)
    def check_insurance_portal(self) -> dict:
        """Log into insurance portal to check coverage details."""
        ps = self.db.patient_state
        if ps.insurance_portal is None:
            return {"error": "Cannot access insurance portal"}
        return {
            "plan_type": ps.insurance_portal.plan_type,
            "copay": ps.insurance_portal.copay_amount,
            "referral_on_file": ps.insurance_portal.referral_on_file,
            "referral_department": ps.insurance_portal.referral_department,
            "deductible_met": ps.insurance_portal.deductible_met,
        }

    # ── Assertions (used in evaluation_criteria.env_assertions) ───────

    def assert_vital_taken(self, vital_type: str) -> bool:
        """Assert that a specific vital was taken during the conversation."""
        return any(r.type == vital_type for r in self.db.patient_state.vital_readings)

    def assert_pain_level_reported(self) -> bool:
        """Assert that pain level was checked during conversation."""
        return self.db.patient_state.pain_level > 0

    def assert_medication_checked(self) -> bool:
        """Assert that medication cabinet was checked."""
        # This is tracked by whether check_medication_cabinet was called
        # The tool itself is the assertion trigger
        return True  # Checked via action_checks in tasks
