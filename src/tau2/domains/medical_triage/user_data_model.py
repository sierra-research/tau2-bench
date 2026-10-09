"""User-side data models for the medical triage domain (patient state)."""

from typing import List, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class VitalReading(BaseModel):
    """A vital sign reading taken by the patient."""

    type: str = Field(
        description="Type of reading (temperature, blood_pressure, pulse, oxygen)"
    )
    value: str = Field(description="The reading value")
    timestamp: str = Field(default="", description="When the reading was taken")
    unit: str = Field(default="", description="Unit of measurement")


class HomeMedication(BaseModel):
    """A medication the patient has at home."""

    name: str = Field(description="Medication name")
    dosage: str = Field(default="", description="Dosage on label")
    quantity_remaining: int = Field(default=0, description="Pills/doses remaining")
    expired: bool = Field(default=False, description="Whether medication is expired")


class InsurancePortalInfo(BaseModel):
    """Information from the patient's insurance portal."""

    plan_type: str = Field(default="", description="Insurance plan type")
    copay_amount: float = Field(default=0.0, description="Copay for visit type")
    referral_on_file: bool = Field(
        default=False, description="Whether a referral exists"
    )
    referral_department: str = Field(
        default="", description="Department referral is for"
    )
    deductible_met: bool = Field(default=True, description="Whether deductible is met")


class PatientDevice(BaseModel):
    """Home medical devices and their availability."""

    has_thermometer: bool = Field(default=True, description="Has a thermometer")
    has_bp_monitor: bool = Field(
        default=False, description="Has a blood pressure monitor"
    )
    has_pulse_oximeter: bool = Field(default=False, description="Has a pulse oximeter")
    has_glucose_meter: bool = Field(default=False, description="Has a glucose meter")


class PatientState(BaseModel):
    """The patient's current state — information only the patient knows."""

    patient_id: str = Field(description="Patient ID for cross-reference")
    current_symptoms: List[str] = Field(
        default_factory=list, description="Current symptoms"
    )
    pain_level: int = Field(default=0, description="Current pain level 1-10")
    symptom_duration_hours: float = Field(
        default=0, description="How long symptoms have lasted"
    )
    temperature_f: Optional[float] = Field(
        default=None, description="Current temperature if measured"
    )
    blood_pressure: Optional[str] = Field(
        default=None, description="Current BP if measured (e.g. 140/90)"
    )
    pulse: Optional[int] = Field(default=None, description="Current pulse if measured")
    oxygen_saturation: Optional[int] = Field(
        default=None, description="SpO2 if measured"
    )
    home_medications: List[HomeMedication] = Field(
        default_factory=list, description="Meds at home"
    )
    insurance_portal: Optional[InsurancePortalInfo] = Field(
        default=None, description="Insurance portal data"
    )
    devices: PatientDevice = Field(
        default_factory=PatientDevice, description="Available devices"
    )
    vital_readings: List[VitalReading] = Field(
        default_factory=list, description="Readings taken during call"
    )


class MedicalUserDB(DB):
    """User-side database for the medical triage domain."""

    patient_state: PatientState = Field(description="Patient's current state")

    def get_statistics(self) -> dict:
        return {
            "symptoms": len(self.patient_state.current_symptoms),
            "home_medications": len(self.patient_state.home_medications),
            "vital_readings": len(self.patient_state.vital_readings),
            "has_devices": any(
                [
                    self.patient_state.devices.has_thermometer,
                    self.patient_state.devices.has_bp_monitor,
                    self.patient_state.devices.has_pulse_oximeter,
                ]
            ),
        }
