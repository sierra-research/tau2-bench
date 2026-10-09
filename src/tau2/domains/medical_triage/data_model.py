"""Data models for the medical triage domain — matched to db.json schema."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tau2.domains.medical_triage.utils import MEDICAL_DB_PATH
from tau2.environment.db import DB


class Insurance(BaseModel):
    type: str = Field(description="Insurance type (medicare, hmo, ppo, medicaid)")
    plan_id: str = Field(default="", description="Plan ID")
    referral_required: bool = Field(
        default=False, description="Whether referrals needed"
    )


class Medication(BaseModel):
    name: str = Field(description="Medication name")
    dosage: str = Field(default="", description="Dosage")
    frequency: str = Field(default="", description="Frequency")


class MedicalHistoryEntry(BaseModel):
    condition: str = Field(default="", description="Condition name")
    date: str = Field(default="", description="Date")
    status: str = Field(default="active", description="Status")
    notes: Optional[str] = Field(default=None, description="Notes")


class Patient(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    date_of_birth: str = Field(description="DOB YYYY-MM-DD")
    age: int = Field(default=0, description="Age")
    gender: str = Field(default="", description="Gender")
    phone: str = Field(default="", description="Phone")
    email: str = Field(default="", description="Email")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    current_medications: List = Field(default_factory=list, description="Current meds")
    medical_history: List = Field(default_factory=list, description="Medical history")
    insurance: Insurance = Field(
        default_factory=lambda: Insurance(type="unknown", plan_id="")
    )
    appointments: List[str] = Field(default_factory=list, description="Appointment IDs")


class Provider(BaseModel):
    provider_id: str = Field(description="Provider ID")
    name: str = Field(description="Provider name")
    department: str = Field(default="", description="Department")
    specialty: str = Field(default="", description="Specialty")
    clinic: str = Field(default="", description="Clinic name")
    available_slots: List[str] = Field(
        default_factory=list, description="Available slots"
    )


class Appointment(BaseModel):
    appointment_id: str = Field(description="Appointment ID")
    patient_id: str = Field(description="Patient ID")
    provider_id: str = Field(default="", description="Provider ID")
    datetime: str = Field(default="", description="Appointment datetime")
    type: str = Field(default="", description="Appointment type")
    reason: str = Field(default="", description="Reason")
    status: str = Field(default="scheduled", description="Status")
    notes: Optional[str] = Field(default=None, description="Notes")
    triage_level: Optional[str] = Field(default=None, description="Triage level")
    referral_id: Optional[str] = Field(default=None, description="Referral ID")
    reschedule_count: int = Field(default=0, description="Times rescheduled")


class Referral(BaseModel):
    referral_id: str = Field(description="Referral ID")
    patient_id: str = Field(description="Patient ID")
    referring_provider_id: str = Field(default="", description="Referring provider")
    target_department: str = Field(default="", description="Target department")
    reason: str = Field(default="", description="Reason")
    status: str = Field(default="active", description="Status")
    created_date: str = Field(default="", description="Created date")
    expiry_date: str = Field(default="", description="Expiry date")
    used: bool = Field(default=False, description="Whether used")


class MedicalDB(DB):
    """Database for the medical triage domain."""

    patients: Dict[str, Patient] = Field(description="Patients by ID")
    providers: Dict[str, Provider] = Field(description="Providers by ID")
    appointments: Dict[str, Appointment] = Field(description="Appointments by ID")
    referrals: Dict[str, Referral] = Field(description="Referrals by ID")

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_patients": len(self.patients),
            "num_providers": len(self.providers),
            "num_appointments": len(self.appointments),
            "num_referrals": len(self.referrals),
        }


def get_db():
    return MedicalDB.load(MEDICAL_DB_PATH)
