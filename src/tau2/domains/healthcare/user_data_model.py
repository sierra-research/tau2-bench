from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB

# Type definitions
SymptomSeverity = Literal["mild", "moderate", "severe"]


class InsuranceCard(BaseModel):
    """Information visible on the patient's insurance card."""

    provider: str = Field(description="Insurance provider name as shown on card")
    policy_number: str = Field(description="Policy number on the card")
    group_number: str = Field(description="Group number on the card")
    member_name: str = Field(description="Member name on the card")
    copay_info: str = Field(description="Copay information printed on card")


class Symptom(BaseModel):
    """A symptom the patient is experiencing."""

    description: str = Field(description="Description of the symptom")
    severity: SymptomSeverity = Field(description="Severity level")
    duration: str = Field(description="How long they've had this symptom")


class MedicationBottle(BaseModel):
    """Information visible on a medication bottle at home."""

    prescription_number: str = Field(description="Prescription number on bottle")
    medication_name: str = Field(description="Name of medication")
    dosage: str = Field(description="Dosage instructions on label")
    refills_remaining: int = Field(description="Number of refills remaining on label")
    prescribing_doctor: str = Field(description="Doctor name on bottle")
    pharmacy_name: str = Field(description="Pharmacy name on bottle")
    pharmacy_phone: str = Field(description="Pharmacy phone number")


class TimeSlot(BaseModel):
    """A time slot in the patient's calendar."""

    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(description="Time in HH:MM format")
    available: bool = Field(description="Whether this slot is available")
    reason: Optional[str] = Field(default=None, description="Reason if not available")


class PatientPortalInfo(BaseModel):
    """Information accessible through the patient portal."""

    upcoming_appointments: List[str] = Field(
        default_factory=list, description="List of upcoming appointment descriptions"
    )
    recent_visits: List[str] = Field(
        default_factory=list, description="List of recent visit summaries"
    )
    test_results_available: bool = Field(
        default=False, description="Whether test results are available to view"
    )
    test_results: List[dict] = Field(
        default_factory=list,
        description="List of test result summaries (test_name, test_date, result, notes)",
    )
    messages_count: int = Field(default=0, description="Number of unread messages")
    outstanding_balance: int = Field(
        default=0, description="Outstanding balance in dollars"
    )


class BloodPressureReading(BaseModel):
    """Blood pressure measurement from home monitor."""

    systolic: int = Field(description="Systolic pressure in mmHg")
    diastolic: int = Field(description="Diastolic pressure in mmHg")


class PainAssessment(BaseModel):
    """PQRST pain assessment."""

    provocation: str = Field(description="What makes it better or worse")
    quality: str = Field(description="Description of pain type")
    radiation: str = Field(description="Where the pain radiates")
    severity: int = Field(description="Pain scale 0-10")
    timing: str = Field(description="When it occurs and duration")


class PatientDevice(BaseModel):
    """
    Represents physical items and information the patient has access to.
    This is what the patient can check or use during the interaction.
    """

    insurance_card: InsuranceCard = Field(description="The patient's insurance card")
    current_symptoms: List[Symptom] = Field(
        default_factory=list,
        description="Symptoms the patient is currently experiencing",
    )
    current_temperature: Optional[float] = Field(
        default=None, description="Current body temperature in Fahrenheit (if measured)"
    )
    medications_at_home: List[MedicationBottle] = Field(
        default_factory=list, description="Medication bottles the patient has at home"
    )
    calendar_availability: List[TimeSlot] = Field(
        default_factory=list, description="Patient's calendar availability"
    )
    portal_info: Optional[PatientPortalInfo] = Field(
        default=None, description="Information from patient portal (if logged in)"
    )

    # Patient actions and confirmations
    confirmed_appointments: List[str] = Field(
        default_factory=list,
        description="List of appointment IDs patient has confirmed",
    )
    consents_provided: List[str] = Field(
        default_factory=list, description="List of consent types patient has provided"
    )
    acknowledged_instructions: List[str] = Field(
        default_factory=list,
        description="List of instruction types patient has acknowledged",
    )
    notification_preferences: List[str] = Field(
        default_factory=list, description="List of enabled notification types"
    )
    pharmacy_transfer_requests: List[dict] = Field(
        default_factory=list, description="List of pharmacy transfer requests"
    )

    # Home medical monitoring devices
    has_blood_pressure_monitor: bool = Field(
        default=False,
        description="Whether patient has a blood pressure monitor at home",
    )
    latest_bp_reading: Optional[BloodPressureReading] = Field(
        default=None, description="Most recent blood pressure reading from home monitor"
    )
    has_glucose_meter: bool = Field(
        default=False, description="Whether patient has a glucose meter at home"
    )
    latest_glucose_reading: Optional[int] = Field(
        default=None, description="Most recent blood glucose reading in mg/dL"
    )
    glucose_measurement_time: Optional[str] = Field(
        default=None,
        description="When glucose was measured (e.g., 'Fasting (8am)', 'After meal')",
    )
    has_pulse_oximeter: bool = Field(
        default=False, description="Whether patient has a pulse oximeter at home"
    )
    latest_spo2_reading: Optional[int] = Field(
        default=None, description="Most recent oxygen saturation (SpO2) percentage"
    )
    latest_heart_rate: Optional[int] = Field(
        default=None, description="Most recent heart rate in beats per minute"
    )
    current_pain: Optional[PainAssessment] = Field(
        default=None, description="Current pain assessment using PQRST format"
    )
    uploaded_photos: List[dict] = Field(
        default_factory=list,
        description="List of photos uploaded during telehealth sessions",
    )


class EmergencyContact(BaseModel):
    """Emergency contact information."""

    name: str = Field(description="Emergency contact's full name")
    phone: str = Field(description="Emergency contact's phone number")
    relationship: str = Field(description="Relationship to patient")


class PatientSurroundings(BaseModel):
    """
    Context and environment around the patient during the interaction.
    """

    patient_id: str = Field(description="The patient's ID in the system")
    full_name: str = Field(description="Patient's full name")
    date_of_birth: str = Field(
        description="Patient's date of birth (for identity verification)"
    )
    location: Literal["home", "work", "on_the_go"] = Field(
        default="home", description="Where the patient is calling from"
    )
    has_internet_access: bool = Field(
        default=True, description="Whether patient has internet access for portal"
    )
    payment_methods_available: List[Literal["credit_card", "debit_card", "cash"]] = (
        Field(
            default_factory=lambda: ["credit_card"],
            description="Payment methods patient has available",
        )
    )
    emergency_contact: Optional[EmergencyContact] = Field(
        default=None, description="Emergency contact information on file"
    )


class HealthcareUserDB(DB):
    """
    Database representing the patient's side of the interaction.
    This contains information the patient can access but the agent cannot see directly.
    """

    patient_device: PatientDevice = Field(
        description="Physical items and information patient can check"
    )
    surroundings: PatientSurroundings = Field(
        description="Context about the patient's current situation"
    )


# Export types for convenience
__all__ = [
    "InsuranceCard",
    "Symptom",
    "MedicationBottle",
    "TimeSlot",
    "PatientPortalInfo",
    "BloodPressureReading",
    "PainAssessment",
    "EmergencyContact",
    "PatientDevice",
    "PatientSurroundings",
    "HealthcareUserDB",
    "SymptomSeverity",
]
